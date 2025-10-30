"""Offline Merlin recommender training script.

This helper can be executed on a workstation that has the NVIDIA Merlin
stack available. It trains (or refreshes) the lightweight embedding model
used by the Streamlit recommendations and persists the resulting
checkpoint so the app can run inference-only.

Usage examples
--------------

Train directly from IGDB (requires TWITCH credentials in env)::

    python streamlit_app/train_merlin_offline.py \
        --max-rows 25000 \
        --output streamlit_app/artifacts/merlin_games.pt

Retrain from an exported analytics dataset::

    python streamlit_app/train_merlin_offline.py \
        --dataset ~/tmp/igdb_games.parquet \
        --output ~/models/merlin_games.pt --force
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from igdb_helpers import get_client, load_games_for_analytics
from recommender import GameRecommender, MERLIN_IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Merlin embeddings for the Streamlit recommender.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("streamlit_app/artifacts/merlin_games.pt"),
        help="Destination path for the Merlin checkpoint. Defaults to streamlit_app/artifacts/merlin_games.pt.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Maximum number of games to pull from IGDB when --dataset is not provided (default: 20000).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional path to a pre-exported analytics dataset (.parquet or .csv). Skips live IGDB pulls when provided.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoints and retrain Merlin from scratch.",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        help="Twitch Client ID override. If omitted, TWITCH_CLIENT_ID env var is used.",
    )
    parser.add_argument(
        "--client-secret",
        type=str,
        help="Twitch Client Secret override. If omitted, TWITCH_CLIENT_SECRET env var is used.",
    )
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset:
        dataset_path = args.dataset.expanduser()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        suffix = dataset_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(dataset_path)
        if suffix == ".csv":
            return pd.read_csv(dataset_path)
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    client_id = args.client_id or os.getenv("TWITCH_CLIENT_ID")
    client_secret = args.client_secret or os.getenv("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Twitch credentials are required. Provide --client-id/--client-secret or set TWITCH_CLIENT_ID/TWITCH_CLIENT_SECRET.")

    client = get_client(client_id, client_secret)
    return load_games_for_analytics(client, max_rows=args.max_rows)


def main() -> None:
    args = parse_args()

    if MERLIN_IMPORT_ERROR:
        raise SystemExit(
            "Merlin dependencies are unavailable. Install `merlin-models` (and GPU drivers if needed).\n"
            f"Import error: {MERLIN_IMPORT_ERROR}"
        )

    df = load_dataset(args)
    if df.empty:
        raise SystemExit("Dataset is empty; cannot train Merlin recommender.")

    checkpoint_path: Optional[str]
    checkpoint_path = str(args.output) if args.output else None

    recommender = GameRecommender(
        df,
        auto_train_merlin=True,
        merlin_checkpoint=checkpoint_path,
        force_merlin_retrain=args.force,
    )

    if recommender.backend != "merlin":
        raise SystemExit(
            "Merlin backend is not active after training. Ensure dependencies are installed and try again with --force."
        )

    saved = recommender.save_merlin_checkpoint(checkpoint_path)
    if not saved:
        raise SystemExit("Failed to persist Merlin checkpoint.")

    print(f"Saved Merlin checkpoint to: {saved}")


if __name__ == "__main__":
    main()
