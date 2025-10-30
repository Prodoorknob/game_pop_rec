# streamlit_app/streamlit_app.py
from __future__ import annotations
import os
import math
from typing import List

import streamlit as st
import pandas as pd
import plotly.express as px

from igdb_helpers import (
    get_client, search_games, fetch_game_details,
    load_games_for_analytics, df_most_rated_genre, df_best_year,
    df_best_platform, df_best_publisher
)
from recommender import GameRecommender, MERLIN_IMPORT_ERROR

st.set_page_config(page_title="Games Popularity Explorer", page_icon="🎮", layout="wide")

# ---------- Secrets / auth ----------
# Put these in .streamlit/secrets.toml (or set env vars in deployment):
# TWITCH_CLIENT_ID = "..."
# TWITCH_CLIENT_SECRET = "..."
CLIENT_ID = st.secrets.get("TWITCH_CLIENT_ID", os.getenv("TWITCH_CLIENT_ID", ""))
CLIENT_SECRET = st.secrets.get("TWITCH_CLIENT_SECRET", os.getenv("TWITCH_CLIENT_SECRET", ""))
if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Missing Twitch credentials. Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in Streamlit secrets.")
    st.stop()

# ---------- Caching ----------
@st.cache_data(show_spinner=False, ttl=60*30)
def _client():
    return get_client(CLIENT_ID, CLIENT_SECRET)

@st.cache_data(show_spinner=True, ttl=60*10)
def _search(q: str):
    return search_games(_client(), q, limit=25)

@st.cache_data(show_spinner=True, ttl=60*60)
def _details(game_id: int):
    return fetch_game_details(_client(), game_id)

@st.cache_data(show_spinner=True, ttl=60*60)
def _analytics_df(max_rows: int = 20000):
    return load_games_for_analytics(_client(), max_rows=max_rows)


@st.cache_resource(show_spinner=True, ttl=60*60)
def _recommender(max_rows: int = 20000) -> GameRecommender:
    df = _analytics_df(max_rows=max_rows)
    return GameRecommender(df)

# ---------- UI ----------
st.title("Games Popularity Explorer")

# Initialize session state for search handling
session = st.session_state
session.setdefault("search_query", "")
session.setdefault("search_results", [])
session.setdefault("search_message", "")
session.setdefault("search_message_type", "info")
session.setdefault("selected_index", 0)

st.markdown(
    """
    <style>
    div[data-testid="stForm"] form {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

search_left, search_center, search_right = st.columns([1, 2, 1])
with search_center:
    with st.form("game_search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Find a game",
            value=session["search_query"],
            placeholder="e.g., The Witcher 3, Celeste, Elden Ring",
            label_visibility="collapsed",
        )
        search_submit = st.form_submit_button("Search", use_container_width=True)

    if search_submit:
        query_clean = query_input.strip()
        if not query_clean:
            session["search_results"] = []
            session["search_message"] = "Enter a search term to get started."
            session["search_message_type"] = "warning"
        else:
            with st.spinner("Searching IGDB..."):
                results = _search(query_clean)
            session["search_query"] = query_clean
            session["search_results"] = results
            session["selected_index"] = 0
            if results:
                session["search_message"] = ""
            else:
                session["search_message"] = "No matches. Try a different title."
                session["search_message_type"] = "info"

    if session.get("search_message"):
        msg = session["search_message"]
        if session.get("search_message_type") == "warning":
            st.warning(msg)
        else:
            st.info(msg)

col_main, col_side = st.columns([7, 5], gap="large")

with col_side:
    st.subheader("Quick questions")
    sample_n = st.slider(
        "Sample size (top by rating count)",
        min_value=5000,
        max_value=40000,
        step=5000,
        value=20000,
    )
    st.caption(
        "Quick visuals powered by IGDB ratings. Increase the sample size above if you need more coverage."
    )
    analytics_df = _analytics_df(max_rows=sample_n)

    q = st.radio(
        "Choose a question",
        options=[
            "Which is the most rated genre?",
            "Which year has the highest rated games?",
            "Which platform has the best games?",
            "Which publisher has the best games?",
        ],
        index=0,
    )

    if q == "Which is the most rated genre?":
        out = df_most_rated_genre(_client(), analytics_df).head(20)
        fig = px.bar(
            out,
            x="avg_rating",
            y="genre",
            orientation="h",
            title="Average rating by genre (top 20)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif q == "Which year has the highest rated games?":
        out = df_best_year(_client(), analytics_df)
        fig = px.line(
            out,
            x="year",
            y="avg_rating",
            markers=True,
            title="Average rating by release year (n≥20)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif q == "Which platform has the best games?":
        out = df_best_platform(_client(), analytics_df).head(30)
        fig = px.bar(
            out,
            x="avg_rating",
            y="platform",
            orientation="h",
            title="Average rating by platform (top 30)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif q == "Which publisher has the best games?":
        out = df_best_publisher(_client(), analytics_df).head(25)
        fig = px.bar(
            out,
            x="avg_rating",
            y="publisher",
            orientation="h",
            title="Average rating by publisher (n≥10)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

with col_main:
    st.subheader("Game search & details")

    results: List[dict] = session.get("search_results", [])
    if results:

        def _label(r):
            year = ""
            if r.get("first_release_date"):
                year = pd.to_datetime(r["first_release_date"], unit="s", errors="coerce").year
                if not (isinstance(year, int) and year > 0):
                    year = ""
            rating = r.get("total_rating")
            rtxt = (
                f" ({rating:.1f})"
                if isinstance(rating, (int, float)) and not math.isnan(rating)
                else ""
            )
            return f"{r.get('name', '')}{f' [{year}]' if year else ''}{rtxt}"

        default_idx = min(session.get("selected_index", 0), len(results) - 1)
        selected_index = st.selectbox(
            "Select a game",
            options=list(range(len(results))),
            format_func=lambda i: _label(results[i]),
            index=default_idx,
            key="selected_game_index",
        )
        session["selected_index"] = selected_index

        selected = results[selected_index]
        details = _details(int(selected["id"]))

        imgs: List[str] = details.get("images", [])
        if imgs:
            st.image(imgs, use_column_width=True, caption=[details["name"]] * len(imgs))

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric("Total rating", f"{details['ratings'].get('total_rating') or '—'}")
            st.metric(
                "Total rating count",
                f"{details['ratings'].get('total_rating_count') or '—'}",
            )
        with meta_cols[1]:
            st.metric(
                "Agg. rating",
                f"{details['ratings'].get('aggregated_rating') or '—'}",
            )
            st.metric(
                "Agg. rating count",
                f"{details['ratings'].get('aggregated_rating_count') or '—'}",
            )
        with meta_cols[2]:
            year = pd.to_datetime(details.get("first_release_date"), unit="s", errors="coerce").year
            st.metric("First release year", year if pd.notna(year) else "—")

        st.markdown("**Genres:** " + (", ".join(details.get("genres", [])) or "—"))
        st.markdown("**Platforms:** " + (", ".join(details.get("platforms", [])) or "—"))
        if details.get("publishers"):
            st.markdown("**Publishers:** " + ", ".join(details["publishers"]))
        if details.get("developers"):
            st.markdown("**Developers:** " + ", ".join(details["developers"]))

        if details.get("summary"):
            with st.expander("Summary", expanded=True):
                st.write(details["summary"])
        if details.get("storyline"):
            with st.expander("Storyline"):
                st.write(details["storyline"])

        sites = details.get("websites", [])
        if sites:
            st.markdown("**Links:**")
            for url, cat in sites:
                st.markdown(f"- [{url}]({url})")

        st.markdown("---")
        st.markdown("#### Recommended for you")

        try:
            recommender = _recommender(sample_n)
        except Exception as exc:
            st.warning(f"Recommendation engine unavailable: {exc}")
        else:
            if not recommender.is_ready:
                message = recommender.warning or "Not enough data to generate recommendations yet."
                st.info(message)
                if MERLIN_IMPORT_ERROR:
                    st.caption(
                        "Install `merlin-models` to enable Merlin-powered embeddings."
                    )
            else:
                recommendations = recommender.recommend_from_details(
                    details,
                    top_k=5,
                    exclude_ids={int(selected["id"])},
                )
                if not recommendations:
                    message = recommender.warning or "Unable to compute recommendations for this game."
                    st.info(message)
                else:
                    st.caption(f"Recommendation engine: {recommender.backend_label}")
                    if recommender.warning and recommender.backend != "merlin":
                        st.caption(recommender.warning)

                    for rec in recommendations:
                        meta_bits = []
                        if rec.release_year:
                            meta_bits.append(str(int(rec.release_year)))
                        if rec.total_rating is not None:
                            meta_bits.append(f"Rating {rec.total_rating:.1f}")
                        if rec.total_rating_count is not None:
                            meta_bits.append(f"Votes {int(rec.total_rating_count)}")
                        meta_line = " • ".join(meta_bits) if meta_bits else "Similar profile"
                        st.markdown(
                            f"**{rec.name}**  \n"
                            f"{meta_line}  \n"
                            f"Similarity score: {rec.score:.2f}"
                        )

    else:
        st.info("Use the centered search bar above to find a game.")

st.markdown("---")
st.caption("Built on your IGDB client (OAuth + paging) and registry conventions for endpoints/fields.")
