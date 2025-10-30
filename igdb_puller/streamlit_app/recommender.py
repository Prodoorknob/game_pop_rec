"""Content-based game recommendation utilities for the Streamlit app.

This module builds lightweight feature embeddings from the IGDB analytics
dataset and exposes a simple recommendation API. It attempts to use
Merlin (if installed) to learn representations, and falls back to a
cosine-similarity engine otherwise. The fallback keeps the Streamlit UI
responsive even when Merlin is unavailable in the execution environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MERLIN_IMPORT_ERROR: Optional[str] = None
try:
    # These imports are optional; the app will continue to work without Merlin.
    import torch

    try:
        from merlin.models.torch.outputs.continuous import RegressionOutput
        from merlin.models.torch import Model as MerlinModel
        from merlin.models.torch.inputs.tabular import TabularFeatures
        from merlin.models.torch.blocks.mlp import MLPBlock
        from merlin.schema import ColumnSchema, Schema, Tags

        _MERLIN_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - optional dependency
        MERLIN_IMPORT_ERROR = str(exc)
        _MERLIN_AVAILABLE = False
except Exception as exc:  # pragma: no cover - optional dependency
    MERLIN_IMPORT_ERROR = str(exc)
    _MERLIN_AVAILABLE = False


@dataclass
class Recommendation:
    id: int
    name: str
    score: float
    total_rating: Optional[float]
    total_rating_count: Optional[float]
    release_year: Optional[int]


class GameRecommender:
    """Hybrid Merlin / cosine similarity recommender.

    Parameters
    ----------
    df : pandas.DataFrame
        Result of ``load_games_for_analytics`` with at least the columns:
        ``id``, ``name``, ``total_rating``, ``total_rating_count``,
        ``genres``, ``platforms``, ``first_release_date``.
    """

    def __init__(self, df: pd.DataFrame):
        self.backend: str = "similarity"
        self.warning: Optional[str] = None
        self._ready: bool = False

        self._ids: List[int] = []
        self._meta: Dict[int, Dict[str, Optional[float]]] = {}
        self._feature_matrix: Optional[np.ndarray] = None

        self._genre_index: Dict[int, int] = {}
        self._platform_index: Dict[int, int] = {}
        self._numeric_stats: Dict[str, Tuple[float, float]] = {}

        self._fit(df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._ready and self._feature_matrix is not None and len(self._ids) > 0

    @property
    def backend_label(self) -> str:
        if self.backend == "merlin":
            return "Merlin two-tower embeddings"
        return "feature similarity (Merlin fallback)"

    def recommend_from_details(
        self,
        details: Dict,
        *,
        top_k: int = 5,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> List[Recommendation]:
        """Recommend games using the fetched game details dictionary."""

        record = self._record_from_details(details)
        if record is None:
            return []
        return self._recommend(record, top_k=top_k, exclude_ids=exclude_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fit(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            self.warning = "Analytics dataset is empty; recommendations are disabled."
            return

        relevant_cols = {
            "id",
            "name",
            "total_rating",
            "total_rating_count",
            "genres",
            "platforms",
            "first_release_date",
        }
        missing = relevant_cols.difference(df.columns)
        if missing:
            self.warning = f"Dataset missing columns: {', '.join(sorted(missing))}."
            return

        df_proc = df.dropna(subset=["id", "name"]).copy()
        if df_proc.empty:
            self.warning = "No analysable games in dataset after filtering."
            return

        df_proc["id"] = df_proc["id"].astype(int)

        # Build vocabularies
        genre_values = self._collect_unique(df_proc["genres"])
        platform_values = self._collect_unique(df_proc["platforms"])
        self._genre_index = {gid: idx for idx, gid in enumerate(sorted(genre_values))}
        self._platform_index = {
            pid: idx + len(self._genre_index)
            for idx, pid in enumerate(sorted(platform_values))
        }

        # Numeric stats for scaling
        ratings = self._safe_series(df_proc.get("total_rating"))
        rating_counts = self._safe_series(df_proc.get("total_rating_count"))
        years = self._safe_series(
            df_proc.get("first_release_date").apply(self._to_year)  # type: ignore[arg-type]
        )

        self._numeric_stats = {
            "total_rating": self._minmax_tuple(ratings),
            "total_rating_count": self._minmax_tuple(np.log1p(rating_counts.dropna())),
            "release_year": self._minmax_tuple(years),
        }

        vectors: List[np.ndarray] = []
        ids: List[int] = []
        meta: Dict[int, Dict[str, Optional[float]]] = {}

        for row in df_proc.itertuples(index=False):
            game_id = int(getattr(row, "id"))
            genres = self._ensure_int_list(getattr(row, "genres", None))
            platforms = self._ensure_int_list(getattr(row, "platforms", None))
            total_rating = self._safe_float(getattr(row, "total_rating", None))
            total_rating_count = self._safe_float(getattr(row, "total_rating_count", None))
            release_year = self._to_year(getattr(row, "first_release_date", None))

            vector = self._build_vector(
                genres=genres,
                platforms=platforms,
                total_rating=total_rating,
                total_rating_count=total_rating_count,
                release_year=release_year,
            )

            if vector is None:
                continue

            vectors.append(vector)
            ids.append(game_id)
            meta[game_id] = {
                "name": getattr(row, "name", ""),
                "total_rating": total_rating,
                "total_rating_count": total_rating_count,
                "release_year": release_year,
            }

        if not vectors:
            self.warning = "Could not build feature vectors for recommendations."
            return

        feature_matrix = np.vstack(vectors)
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feature_matrix = feature_matrix / norms

        self._feature_matrix = feature_matrix
        self._ids = ids
        self._meta = meta
        self._ready = True

        # Attempt optional Merlin training — if it fails, we keep the fallback.
        if _MERLIN_AVAILABLE:
            try:
                self._train_merlin(feature_matrix)
                self.backend = "merlin"
            except Exception as exc:  # pragma: no cover - optional dependency
                self.warning = (
                    "Merlin training fallback engaged: "
                    + (str(exc) or "unexpected Merlin error")
                )

    # ------------------------------------------------------------------
    def _train_merlin(self, feature_matrix: np.ndarray) -> None:
        """Train a simple Merlin MLP to re-embed item features.

        The model learns to reconstruct the cosine-projected features,
        yielding smoother embeddings for similarity search. This method is
        only executed when Merlin is installed. If anything goes wrong we
        re-raise to trigger the fallback warning upstream.
        """

        # Build a trivial schema with a single continuous feature block.
        feature_dim = feature_matrix.shape[1]
        schema = Schema([
            ColumnSchema(
                "features",
                properties={"shape": (feature_dim,)},
                tags=(Tags.CONTINUOUS,),
            )
        ])

        inputs = TabularFeatures(schema)
        tower = MLPBlock([feature_dim, feature_dim], activations="relu")
        model = MerlinModel(inputs, tower, RegressionOutput())

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(feature_matrix, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for _ in range(5):  # a few rapid epochs suffice for smoothing
            for batch in loader:
                optim.zero_grad()
                outputs = model(batch[0])
                loss = torch.nn.functional.mse_loss(outputs, batch[0])
                loss.backward()
                optim.step()

        model.eval()
        with torch.no_grad():
            refined = model(torch.tensor(feature_matrix, dtype=torch.float32)).numpy()

        norms = np.linalg.norm(refined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._feature_matrix = refined / norms

    # ------------------------------------------------------------------
    def _recommend(
        self,
        record: Dict[str, Optional[float]],
        *,
        top_k: int,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> List[Recommendation]:
        if not self.is_ready or self._feature_matrix is None:
            return []

        vector = self._build_vector(
            genres=self._ensure_int_list(record.get("genres")),
            platforms=self._ensure_int_list(record.get("platforms")),
            total_rating=self._safe_float(record.get("total_rating")),
            total_rating_count=self._safe_float(record.get("total_rating_count")),
            release_year=self._to_year(record.get("release_year")),
        )

        if vector is None:
            return []

        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            return []
        vector = vector / vector_norm

        similarities = self._feature_matrix @ vector
        exclude = set(int(x) for x in (exclude_ids or []))

        # Retrieve top-k indices
        ranking = np.argsort(similarities)[::-1]
        recommendations: List[Recommendation] = []
        for idx in ranking:
            game_id = self._ids[idx]
            if game_id in exclude:
                continue
            info = self._meta.get(game_id, {})
            recommendations.append(
                Recommendation(
                    id=game_id,
                    name=str(info.get("name", "")),
                    score=float(similarities[idx]),
                    total_rating=self._safe_float(info.get("total_rating")),
                    total_rating_count=self._safe_float(info.get("total_rating_count")),
                    release_year=self._to_year(info.get("release_year")),
                )
            )
            if len(recommendations) >= top_k:
                break

        return recommendations

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _build_vector(
        self,
        *,
        genres: Sequence[int],
        platforms: Sequence[int],
        total_rating: Optional[float],
        total_rating_count: Optional[float],
        release_year: Optional[int],
    ) -> Optional[np.ndarray]:
        dim = len(self._genre_index) + len(self._platform_index) + 3
        if dim <= 0:
            return None

        vector = np.zeros(dim, dtype=np.float32)

        for gid in genres:
            pos = self._genre_index.get(int(gid))
            if pos is not None:
                vector[pos] = 1.0

        for pid in platforms:
            pos = self._platform_index.get(int(pid))
            if pos is not None:
                vector[pos] = 1.0

        offset = len(self._genre_index) + len(self._platform_index)
        vector[offset] = self._scale_numeric(total_rating, "total_rating")
        vector[offset + 1] = self._scale_numeric(
            None if total_rating_count is None else np.log1p(total_rating_count),
            "total_rating_count",
        )
        vector[offset + 2] = self._scale_numeric(release_year, "release_year")

        return vector

    def _record_from_details(self, details: Dict) -> Optional[Dict[str, Optional[float]]]:
        if not details:
            return None

        raw = details.get("raw", {}) or {}
        ratings = details.get("ratings", {}) or {}

        record: Dict[str, Optional[float]] = {
            "genres": self._ensure_int_list(raw.get("genres")),
            "platforms": self._ensure_int_list(raw.get("platforms")),
            "total_rating": self._safe_float(ratings.get("total_rating")),
            "total_rating_count": self._safe_float(ratings.get("total_rating_count")),
            "release_year": self._to_year(details.get("first_release_date")),
        }

        # Fall back to aggregated counts in raw payload.
        if record["total_rating"] is None:
            record["total_rating"] = self._safe_float(raw.get("total_rating"))
        if record["total_rating_count"] is None:
            record["total_rating_count"] = self._safe_float(raw.get("total_rating_count"))
        if record["release_year"] is None:
            record["release_year"] = self._to_year(raw.get("first_release_date"))

        return record

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_unique(series: pd.Series) -> List[int]:
        values: List[int] = []
        if series is None:
            return values
        for item in series.dropna():
            values.extend(GameRecommender._ensure_int_list(item))
        return sorted(set(values))

    @staticmethod
    def _ensure_int_list(value) -> List[int]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
            out: List[int] = []
            for v in value:
                try:
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    out.append(int(v))
                except Exception:
                    continue
            return out
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, str):
            tokens = [t.strip() for t in value.replace("|", ",").split(",") if t.strip()]
            out = []
            for token in tokens:
                try:
                    out.append(int(float(token)))
                except Exception:
                    continue
            return out
        try:
            return [int(value)]
        except Exception:
            return []

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            if isinstance(value, float) and math.isnan(value):
                return None
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_series(series: Optional[pd.Series]) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def _to_year(value) -> Optional[int]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        try:
            ts = pd.to_datetime(value, unit="s", errors="coerce")
        except Exception:
            try:
                ts = pd.to_datetime(value, errors="coerce")
            except Exception:
                ts = pd.NaT
        if pd.isna(ts):
            return None
        year = ts.year
        return int(year) if not pd.isna(year) else None

    @staticmethod
    def _minmax_tuple(series: pd.Series) -> Tuple[float, float]:
        if series is None or series.dropna().empty:
            return (0.0, 0.0)
        s = series.dropna()
        return (float(s.min()), float(s.max()))

    def _scale_numeric(self, value: Optional[float], key: str) -> float:
        stats = self._numeric_stats.get(key)
        if stats is None:
            return 0.0
        lo, hi = stats
        if value is None or math.isnan(value):
            return 0.0
        if hi <= lo:
            return 0.0
        clipped = float(min(max(value, lo), hi))
        return (clipped - lo) / (hi - lo)

