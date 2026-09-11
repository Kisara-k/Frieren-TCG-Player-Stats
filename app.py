import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import subprocess
from datetime import datetime, timedelta
import rankings as _rankings
from matchup_chart import make_character_matchup_chart

st.set_page_config(page_title="Frieren TCG Player Stats and Leaderboard", layout="wide")

_primary_color = st.get_option("theme.primaryColor") or "#4fa3d1"
_pc = _primary_color.lstrip("#")
_primary_rgba = "rgba({},{},{},0.35)".format(int(_pc[0:2], 16), int(_pc[2:4], 16), int(_pc[4:6], 16))

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- DATA LOADING --------------------------------------------------------------
def _get_data_stamp() -> tuple[str, str]:
    """Return (cache_stamp, display_str) from Match.csv's last-modified date.
    The stamp is passed to load_data() so the cache invalidates when data changes."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ai", "data/Match.csv"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            full_str = result.stdout.strip()
            date_str = full_str.split()[0]
            return full_str, datetime.fromisoformat(date_str).strftime('%B %d, %Y')
    except Exception:
        pass
    try:
        if os.path.exists("data/Match.csv"):
            mtime = os.path.getmtime("data/Match.csv")
            return str(mtime), datetime.fromtimestamp(mtime).strftime('%B %d, %Y')
    except Exception:
        pass
    return "unknown", "Unknown"

_data_stamp, _last_updated_str = _get_data_stamp()

@st.cache_data
def load_data(stamp: str):  # stamp unused inside; changing it busts the cache
    print(f"[cache MISS] load_data() called with stamp={stamp!r}")
    players = pd.read_csv("data/Player.csv")
    matches = pd.read_csv("data/Match.csv")
    characters = pd.read_csv("data/Character.csv")
    ladder_resets = pd.read_csv("data/LadderReset.csv")
    ladders = pd.read_csv("data/Ladder.csv")
    players["name"] = players["name"].replace("", pd.NA).fillna(players["discordName"])
    return players, matches, characters, ladder_resets, ladders

players, matches, characters, ladder_resets, ladders = load_data(_data_stamp)

# ladderResetId -> ladder name (e.g. "classic", "blitz", "slow", "classic-prescience")
_reset_to_ladder_name = (
    ladder_resets.merge(ladders, left_on="ladderId", right_on="id")[["id_x", "name"]]
    .rename(columns={"id_x": "ladderResetId"})
    .set_index("ladderResetId")["name"]
    .to_dict()
)

# ladderResetId -> canonical season label (S1, S2, ...) based on classic ladder seasons.
# Every non-classic reset is assigned to whichever classic season was active at the same time
# (i.e. the latest classic reset whose startDate is <= this reset's startDate).
_classic_resets = (
    ladder_resets[ladder_resets["ladderId"] == 1]
    .sort_values("startDate")
    .reset_index(drop=True)
    .assign(season_num=lambda df: range(1, len(df) + 1))
)
_reset_to_season = {}
for _, _r in ladder_resets.iterrows():
    _mask = _classic_resets["startDate"] <= _r["startDate"]
    _snum = int(_classic_resets.loc[_mask, "season_num"].iloc[-1]) if _mask.any() else 1
    _reset_to_season[int(_r["id"])] = f"S{_snum}"

# season label -> start timestamp (ms) - used for weekly breakdown
_season_start_ms = {
    f"S{int(row['season_num'])}": int(row["startDate"])
    for _, row in _classic_resets.iterrows()
}

def _build_player_label(row) -> str:
    if "name" in players.columns and pd.notna(row.get("name")) and str(row.get("name", "")).strip():
        return str(row["name"])
    if pd.notna(row.get("discordName")) and str(row.get("discordName", "")).strip():
        return str(row["discordName"])
    return str(row["discordId"])

player_label_map: dict[int, str] = {
    row["id"]: _build_player_label(row) for _, row in players.iterrows()
}
for _pid in pd.concat([matches["winnerId"], matches["loserId"]]).unique():
    if _pid not in player_label_map:
        player_label_map[_pid] = f"player_{_pid}"

char_map = characters.set_index("id")["name"].to_dict()
char_color_map = characters.set_index("name")["Hex"].to_dict() if "Hex" in characters.columns else {}


@st.cache_data
def build_player_matches(discord_id_str: str):
    """Build the per-match DataFrame for a given Discord ID string.
    Cached by @st.cache_data so it only recomputes when the ID changes.
    discord_id_str must be a string so cache keys are consistent."""
    row = players[players["discordId"] == int(discord_id_str)]
    if row.empty:
        return None, None, None

    player_id = int(row["id"].values[0])
    player_label = player_label_map[player_id]

    as_winner = matches[matches["winnerId"] == player_id].copy()
    as_winner["result"] = "Win"
    as_winner["player_char"] = as_winner["winnerCharacterId"]
    as_winner["opp_id"] = as_winner["loserId"]
    as_winner["opp_char"] = as_winner["loserCharacterId"]

    as_loser = matches[matches["loserId"] == player_id].copy()
    as_loser["result"] = "Loss"
    as_loser["player_char"] = as_loser["loserCharacterId"]
    as_loser["opp_id"] = as_loser["winnerId"]
    as_loser["opp_char"] = as_loser["winnerCharacterId"]

    pm = pd.concat([as_winner, as_loser], ignore_index=True)
    pm["season"] = pm["ladderResetId"].map(_reset_to_season)
    pm["player_char_name"] = pm["player_char"].map(char_map)
    pm["opp_char_name"] = pm["opp_char"].map(char_map)
    pm["opp_label"] = pm["opp_id"].map(player_label_map)
    # Normalise ranked: null → 0 (unranked), any non-null value → 1 (ranked)
    pm["ranked"] = pm["ranked"].notna().astype(int)
    pm["ladder_name"] = pm["ladderResetId"].map(_reset_to_ladder_name)

    # Pre-compute season game counts (stable reference for dropdown labels)
    season_counts = (
        pm.groupby("season")["result"]
        .count()
        .reindex(sorted(pm["season"].unique(), key=lambda s: int(s[1:])))
        .to_dict()
    )

    return pm, player_label, season_counts


def matchup_stats(df, group_col="opp_label", top_n=None):
    stats = df.groupby(group_col).agg(
        Games=("result", "count"),
        Wins=("result", lambda x: (x == "Win").sum()),
        Losses=("result", lambda x: (x == "Loss").sum()),
    ).reset_index()
    stats["Games"] = stats["Games"].astype(int)
    stats["Wins"] = stats["Wins"].astype(int)
    stats["Losses"] = stats["Losses"].astype(int)
    stats["WinRate"] = (stats["Wins"] / stats["Games"] * 100).round(1)
    stats = stats.sort_values("Games", ascending=False)
    if top_n:
        stats = stats.head(top_n)
    return stats


def _hex_to_hsl_capped(hex_color: str, max_l: int = 60) -> str:
    """Convert hex color to hsl() CSS string with lightness capped at max_l."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    s = 0.0 if delta == 0 else delta / (1 - abs(2 * l - 1))
    if delta == 0:
        hue = 0.0
    elif cmax == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        hue = 60 * ((b - r) / delta + 2)
    else:
        hue = 60 * ((r - g) / delta + 4)
    l_capped = min(l, max_l / 100)
    if l > l_capped:
        s = s * (l_capped / l)
    return f"hsl({hue:.0f},{s * 100:.0f}%,{l_capped * 100:.0f}%)"


def char_picks_str(df_col: pd.Series, top_n: int = 4) -> str:
    """Return top-N character pick distribution as a colored HTML string for tooltips."""
    counts = df_col.dropna().value_counts()
    total = counts.sum()
    if total == 0:
        return "-"
    parts = []
    for c in counts.index[:top_n]:
        color = _hex_to_hsl_capped(char_color_map.get(c, "#AAAAAA"))
        pct = f"{counts[c] / total * 100:.0f}%"
        parts.append(f'<span style="color:{color}"><b>{c}</b> {pct}</span>')
    return "  ·  ".join(parts)


# -- TABS ----------------------------------------------------------------------
st.title("Frieren TCG Player Stats and Leaderboard")
st.caption(f"Last updated: {_last_updated_str}")

_main_view = st.radio(
    "View", ["Player Stats", "Leaderboard"],
    horizontal=True, key="main_view", label_visibility="collapsed",
)

# -- VIEW: PLAYER STATS --------------------------------------------------------
if _main_view == "Player Stats":
    # Map player display name -> discord ID string (built once from cached data)
    name_to_discord_id: dict[str, str] = {
        label: str(players.loc[players["id"] == pid, "discordId"].values[0])
        for pid, label in player_label_map.items()
    }
    all_names_sorted = sorted(name_to_discord_id.keys())

    # on_change callbacks - fired immediately on name selection, or on Enter/blur for text input
    def _commit_by_name():
        name = st.session_state.get("player_name_select")
        if name:
            st.session_state["confirmed_discord_id"] = name_to_discord_id[name]

    def _commit_by_id():
        raw = st.session_state.get("discord_id_input", "").strip()
        if not raw:
            return
        if raw.isdigit():
            try:
                row = players[players["discordId"] == int(raw)]
            except OverflowError:
                row = pd.DataFrame()
            if not row.empty:
                st.session_state["confirmed_discord_id"] = raw

    col_name, col_id, col_btn = st.columns([4, 3, 1])
    with col_name:
        st.selectbox(
            "Player name",
            options=all_names_sorted,
            index=None,
            placeholder="Type to search player name…",
            label_visibility="collapsed",
            key="player_name_select",
            on_change=_commit_by_name,
        )
    with col_id:
        st.text_input(
            "Discord ID",
            placeholder="Or paste Discord ID…",
            label_visibility="collapsed",
            key="discord_id_input",
            on_change=_commit_by_id,
        )
    with col_btn:
        if st.button("Analyze", type="primary", width="stretch"):
            _raw = st.session_state.get("discord_id_input", "").strip()
            _name = st.session_state.get("player_name_select")
            if _raw and _raw.isdigit():
                _commit_by_id()
            elif _name:
                _commit_by_name()
            else:
                st.warning("Select a player name or enter a Discord ID first.")

    confirmed_id = st.session_state.get("confirmed_discord_id", "")

    if not confirmed_id:
        st.info("Search by name (click and type in the dropdown) or paste a Discord ID and press **Enter**.")
    else:
        # -- LOAD / CACHE PLAYER DATA ------------------------------------------
        _pm_valid = True
        try:
            pm, player_label, season_counts = build_player_matches(confirmed_id)
        except (ValueError, OverflowError):
            st.error("Invalid Discord ID - must be a numeric snowflake.")
            _pm_valid = False

        if _pm_valid:
            if pm is None or pm.empty:
                st.error(f"No player found with Discord ID `{confirmed_id}`.")
            else:
                st.subheader(f"{player_label}")

                # -- SHARED SEASON / LADDER / MATCH-TYPE FILTERS --------------------
                _LADDER_OPTIONS = ["Classic", "All", "Blitz", "Slow", "Prescience"]
                _LADDER_RAW = {"Classic": "classic", "Blitz": "blitz", "Slow": "slow", "Prescience": "classic-prescience", "All": None}

                _count_ladder_mode = st.session_state.get("ladder_filter", "Classic")
                _count_ranked_mode = st.session_state.get("ranked_filter", "All")
                _count_include_self = st.session_state.get("include_self", False)
                _pm_count_context = pm
                _count_ladder_raw = _LADDER_RAW.get(_count_ladder_mode)
                if _count_ladder_raw is not None:
                    _pm_count_context = _pm_count_context[_pm_count_context["ladder_name"] == _count_ladder_raw]
                if _count_ranked_mode == "Ranked":
                    _pm_count_context = _pm_count_context[_pm_count_context["ranked"] == 1]
                elif _count_ranked_mode == "Unranked":
                    _pm_count_context = _pm_count_context[_pm_count_context["ranked"] == 0]
                if not _count_include_self:
                    _pm_count_context = _pm_count_context[_pm_count_context["winnerId"] != _pm_count_context["loserId"]]

                _all_seasons = sorted(pm["season"].dropna().unique(), key=lambda s: int(s[1:]))
                _season_counts = _pm_count_context["season"].value_counts().to_dict()
                _ALL_SEASONS = "All Seasons"

                def _player_season_label(season):
                    games = len(_pm_count_context) if season == _ALL_SEASONS else _season_counts.get(season, 0)
                    return f"{season} ({games:,} games)"

                _player_season_key = f"player_seasons_{confirmed_id}"
                _player_season_previous_key = f"player_seasons_previous_{confirmed_id}"
                if _player_season_key not in st.session_state:
                    st.session_state[_player_season_key] = [_ALL_SEASONS]
                if _player_season_previous_key not in st.session_state:
                    st.session_state[_player_season_previous_key] = list(
                        st.session_state[_player_season_key]
                    )

                def _sync_player_seasons():
                    current = list(st.session_state.get(_player_season_key, []))
                    previous = list(st.session_state.get(_player_season_previous_key, []))
                    if _ALL_SEASONS in current and _ALL_SEASONS not in previous:
                        current = [_ALL_SEASONS]
                    elif _ALL_SEASONS in current and any(s != _ALL_SEASONS for s in current):
                        current = [s for s in current if s != _ALL_SEASONS]
                    st.session_state[_player_season_key] = current
                    st.session_state[_player_season_previous_key] = current

                _col_season, _col_ladder, _col_ranked = st.columns([1.5, 2.5, 2])
                with _col_season:
                    selected_season_options = st.multiselect(
                        "Season", [_ALL_SEASONS] + list(reversed(_all_seasons)),
                        key=_player_season_key, format_func=_player_season_label,
                        on_change=_sync_player_seasons,
                        placeholder="Select seasons", label_visibility="collapsed",
                    )
                with _col_ladder:
                    ladder_mode = st.radio(
                        "Ladder",
                        options=_LADDER_OPTIONS,
                        index=0,
                        horizontal=True,
                        key="ladder_filter",
                        label_visibility="collapsed",
                    )
                with _col_ranked:
                    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                        ranked_mode = st.radio(
                            "Match type", options=["All", "Ranked", "Unranked"],
                            horizontal=True, key="ranked_filter", label_visibility="collapsed",
                            width="content",
                        )
                        include_self = st.checkbox(
                            "Self", value=False, key="include_self", width="content",
                        )
                # Apply ladder filter first, then ranked filter
                _ladder_raw = _LADDER_RAW[ladder_mode]
                if _ladder_raw is not None:
                    pm_ladder = pm[pm["ladder_name"] == _ladder_raw].copy()
                else:
                    pm_ladder = pm

                if ranked_mode == "Ranked":
                    pm_filtered = pm_ladder[pm_ladder["ranked"] == 1].copy()
                elif ranked_mode == "Unranked":
                    pm_filtered = pm_ladder[pm_ladder["ranked"] == 0].copy()
                else:
                    pm_filtered = pm_ladder

                if not include_self:
                    pm_filtered = pm_filtered[pm_filtered["winnerId"] != pm_filtered["loserId"]].copy()

                if not selected_season_options or _ALL_SEASONS in selected_season_options:
                    selected_seasons = sorted(pm_filtered["season"].dropna().unique(), key=lambda s: int(s[1:]))
                    season_label = "All Seasons"
                else:
                    selected_seasons = [s for s in _all_seasons if s in selected_season_options]
                    pm_filtered = pm_filtered[pm_filtered["season"].isin(selected_seasons)].copy()
                    season_label = ", ".join(selected_seasons)

                # Recompute season counts from the filtered set for accurate dropdown labels
                _filtered_season_counts = (
                    pm_filtered.groupby("season")["result"]
                    .count()
                    .reindex(sorted(pm_filtered["season"].unique(), key=lambda s: int(s[1:])) if not pm_filtered.empty else [])
                    .dropna()
                    .astype(int)
                    .to_dict()
                ) if not pm_filtered.empty else {}

                seasons = selected_seasons
                total_games = sum(_filtered_season_counts.values())

                _season_breakdown = "  ·  ".join(f"{s}: {n}" for s, n in _filtered_season_counts.items())
                st.caption(f"Total matches: {total_games}  |  {_season_breakdown}")

                if pm_filtered.empty:
                    st.info("No matches for this selection.")
                    st.stop()

                # -- SECTION 1: OVERALL MATCHUPS ----------------------------------
                st.header("Overall Matchups")

                df_matchup = pm_filtered
                season_label_matchup = season_label

                overall = matchup_stats(df_matchup)

                top20 = overall.head(20).copy()
                top20["MyPicks"] = top20["opp_label"].map(
                    lambda o: char_picks_str(df_matchup[df_matchup["opp_label"] == o]["player_char_name"])
                )
                top20["OppPicks"] = top20["opp_label"].map(
                    lambda o: char_picks_str(df_matchup[df_matchup["opp_label"] == o]["opp_char_name"])
                )
                long20 = top20.melt(
                    id_vars="opp_label", value_vars=["Wins", "Losses"],
                    var_name="Result", value_name="Count",
                )
                long20 = long20.merge(top20[["opp_label", "Games", "WinRate", "Wins", "Losses", "MyPicks", "OppPicks"]], on="opp_label")
                fig1 = px.bar(
                    long20, x="opp_label", y="Count", color="Result",
                    color_discrete_map={"Wins": "#2ecc71", "Losses": "#e74c3c"},
                    title=f"Top 20 Opponents by Games Played - {season_label_matchup} ({player_label})",
                    labels={"opp_label": "Opponent", "Count": "Games"},
                    text_auto=True, barmode="stack",
                    category_orders={"opp_label": top20["opp_label"].tolist()},
                    custom_data=["Games", "WinRate", "Wins", "Losses", "MyPicks", "OppPicks"],
                )
                fig1.update_traces(
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Wins %{customdata[2]}  |  Losses %{customdata[3]}  |  Total %{customdata[0]}  |  WR: %{customdata[1]}%<br>"
                        "<b>Your picks:</b> %{customdata[4]}<br>"
                        "<b>Their picks:</b> %{customdata[5]}"
                        "<extra></extra>"
                    )
                )
                fig1.update_layout(legend_title_text="Result", hoverlabel=dict(align="left"))
                st.plotly_chart(fig1, width='stretch')

                _max_threshold = 50
                _default_min = _max_threshold
                for _threshold in range(1, _max_threshold+1, 2):
                    if len(overall[overall["Games"] >= _threshold]) <= 20:
                        _default_min = _threshold
                        break
                min_games = st.slider("Minimum games for win-rate chart", 1, _max_threshold, _default_min, key=f"min_games_slider_{confirmed_id}_{season_label_matchup}")
                wr_df = overall[overall["Games"] >= min_games].sort_values("WinRate", ascending=True).reset_index(drop=True)
                fig2 = px.scatter(
                    wr_df, x="WinRate", y="opp_label", size="Games", color="WinRate",
                    color_continuous_scale="RdYlGn", range_color=[0, 100],
                    title=f"Win Rate vs Each Opponent (>={min_games} games) - {season_label_matchup} ({player_label})",
                    labels={"opp_label": "Opponent", "WinRate": "Win Rate (%)"},
                    hover_data={"Games": True, "Wins": True, "Losses": True, "WinRate": True},
                )
                fig2.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
                fig2.update_xaxes(range=[-5, 105])
                fig2.update_layout(
                    coloraxis_showscale=False,
                    height=max(420, len(wr_df) * 28 + 120),
                    yaxis={"categoryorder": "array", "categoryarray": wr_df["opp_label"].tolist()},
                )
                st.plotly_chart(fig2, width='stretch')


                # -- SECTION 2: PER-SEASON BREAKDOWN ------------------------------
                st.header("Season & Weekly Breakdown")
                st.caption("A single selected season shows its weekly breakdown; multiple seasons show a season-by-season overview.")

                top_n_season = 6


                def _build_period_overview(df, period_col, period_order):
                    ov = df.groupby(period_col).agg(
                        Games=("result", "count"),
                        Wins=("result", lambda x: (x == "Win").sum()),
                        Losses=("result", lambda x: (x == "Loss").sum()),
                    ).reset_index()
                    ov = ov.set_index(period_col).reindex(period_order).reset_index()
                    ov["WinRate"] = (ov["Wins"].astype(float) / ov["Games"].astype(float) * 100).round(1)
                    ov["MyPicks"] = ov[period_col].map(
                        lambda p: char_picks_str(df[df[period_col] == p]["player_char_name"])
                    )
                    ov["OppPicks"] = ov[period_col].map(
                        lambda p: char_picks_str(df[df[period_col] == p]["opp_char_name"])
                    )
                    return ov


                def _make_overview_chart(ov, period_col, period_order, title):
                    _cd = ov[["WinRate", "Wins", "Losses", "Games", "MyPicks", "OppPicks"]].values
                    _ht = (
                        "<b>%{x}</b><br>"
                        "Wins %{customdata[1]}  |  Losses %{customdata[2]}  |  Total %{customdata[3]}  |  WR: %{customdata[0]}%<br>"
                        "<b>Your picks:</b> %{customdata[4]}<br>"
                        "<b>Their picks:</b> %{customdata[5]}<extra></extra>"
                    )
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(
                        x=ov[period_col], y=ov["Wins"],
                        name="Wins", marker_color="#2ecc71",
                        text=ov["Wins"], textposition="inside",
                        customdata=_cd, hovertemplate=_ht,
                    ), secondary_y=False)
                    fig.add_trace(go.Bar(
                        x=ov[period_col], y=ov["Losses"],
                        name="Losses", marker_color="#e74c3c",
                        text=ov["Losses"], textposition="inside",
                        customdata=_cd, hovertemplate=_ht,
                    ), secondary_y=False)
                    _wr_rows = ov[ov["Games"] >= 10]
                    fig.add_trace(go.Scatter(
                        x=_wr_rows[period_col], y=_wr_rows["WinRate"],
                        name="Win Rate %", mode="lines+markers+text",
                        line=dict(color="#3498db", width=2), marker=dict(size=8),
                        text=_wr_rows["WinRate"].astype(str) + "%", textposition="top center",
                    ), secondary_y=True)
                    fig.update_layout(title=title, barmode="stack", legend_title_text="", hoverlabel=dict(align="left"))
                    fig.update_xaxes(categoryorder="array", categoryarray=period_order)
                    fig.update_yaxes(title_text="Games", secondary_y=False)
                    fig.update_yaxes(title_text="Win Rate (%)", range=[0, 110], secondary_y=True, showgrid=False)
                    return fig


                def _build_period_opp_df(df, period_col, period_order):
                    rows = []
                    for period in period_order:
                        df_p = df[df[period_col] == period]
                        stats_p = matchup_stats(df_p, top_n=top_n_season)
                        for rank, (_, row) in enumerate(stats_p.iterrows(), start=1):
                            rows.append({
                                "period": period,
                                "opp_label": row["opp_label"],
                                "rank": rank,
                                "Wins": int(row["Wins"]),
                                "Losses": int(row["Losses"]),
                                "Games": int(row["Games"]),
                                "WinRate": row["WinRate"],
                            })
                    if not rows:
                        return pd.DataFrame(columns=["period", "opp_label", "rank", "Wins", "Losses", "Games", "WinRate", "MyPicks", "OppPicks"])
                    opp_df = pd.DataFrame(rows)
                    _tmp = df.copy()
                    _tmp["_period"] = _tmp[period_col]
                    picks = (
                        _tmp.groupby(["_period", "opp_label"])[["player_char_name", "opp_char_name"]]
                        .apply(lambda g: pd.Series({
                            "MyPicks": char_picks_str(g["player_char_name"]),
                            "OppPicks": char_picks_str(g["opp_char_name"]),
                        }))
                        .reset_index()
                        .rename(columns={"_period": "period"})
                    )
                    return opp_df.merge(picks, on=["period", "opp_label"], how="left")


                def _make_period_opp_chart(period_opp_df, period_order, x_title, chart_title):
                    df_sub = period_opp_df[period_opp_df["period"].isin(period_order)].copy()
                    fig = go.Figure()
                    wins_in_legend = False
                    losses_in_legend = False
                    for rank in range(1, top_n_season + 1):
                        df_rank = df_sub[df_sub["rank"] == rank].copy()
                        if df_rank.empty:
                            continue
                        og = f"rank{rank}"
                        _cd = df_rank[["opp_label", "Games", "WinRate", "Wins", "Losses", "MyPicks", "OppPicks"]].values
                        _ht = (
                            "<b>%{customdata[0]}</b><br>"
                            "%{x} #" + str(rank) + "<br>"
                            "Wins %{customdata[3]}  |  Losses %{customdata[4]}  |  Total %{customdata[1]}  |  WR: %{customdata[2]}%<br>"
                            "<b>Your picks:</b> %{customdata[5]}<br>"
                            "<b>Their picks:</b> %{customdata[6]}"
                            "<extra></extra>"
                        )
                        fig.add_trace(go.Bar(
                            name="Wins", x=df_rank["period"], y=df_rank["Wins"],
                            marker_color="#2ecc71", legendgroup="Wins",
                            showlegend=not wins_in_legend, offsetgroup=og,
                            customdata=_cd, hovertemplate=_ht,
                        ))
                        wins_in_legend = True
                        fig.add_trace(go.Bar(
                            name="Losses", x=df_rank["period"], y=df_rank["Losses"],
                            base=df_rank["Wins"].tolist(),
                            marker_color="#e74c3c", legendgroup="Losses",
                            showlegend=not losses_in_legend, offsetgroup=og,
                            text=df_rank["opp_label"], textposition="outside",
                            textangle=-90, textfont=dict(size=11),
                            outsidetextfont=dict(size=11), constraintext="none", cliponaxis=False,
                            customdata=_cd, hovertemplate=_ht,
                        ))
                        losses_in_legend = True
                    label_top_margin = 40
                    fig.update_layout(
                        title=chart_title, barmode="group",
                        xaxis=dict(title=x_title, categoryorder="array", categoryarray=period_order),
                        yaxis_title="Games", legend_title_text="",
                        height=420 + label_top_margin,
                        margin=dict(t=label_top_margin, b=60, l=60, r=20),
                        bargap=0.15, bargroupgap=0.05,
                        hoverlabel=dict(align="left"),
                    )
                    return fig


                _show_season_overview = (
                    not selected_season_options
                    or _ALL_SEASONS in selected_season_options
                    or len(selected_seasons) != 1
                )
                if _show_season_overview:
                    _sov_order = selected_seasons
                    season_overview = _build_period_overview(pm_filtered, "season", _sov_order)
                    st.plotly_chart(
                        _make_overview_chart(season_overview, "season", _sov_order, f"Season Overview - {player_label}"),
                        width='stretch',
                    )
                    season_opp_df = _build_period_opp_df(pm_filtered, "season", _sov_order)
                    st.plotly_chart(
                        _make_period_opp_chart(season_opp_df, _sov_order, "Season", f"Top {top_n_season} Opponents per Season - {player_label}"),
                        width='stretch',
                    )

                else:
                    _sel_season = selected_seasons[0]
                    _df_week = pm_filtered.copy()

                    if _df_week.empty:
                        st.info(f"No matches found for {_sel_season}.")
                    else:
                        _season_start_ms_val = _season_start_ms.get(_sel_season, int(_df_week["finishedAt"].min()))
                        _anchor_dt = datetime.utcfromtimestamp(_season_start_ms_val / 1000)
                        _anchor_monday = _anchor_dt - timedelta(days=_anchor_dt.weekday())
                        _anchor_monday_ms = int(_anchor_monday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
                        _ms_per_week = 7 * 24 * 60 * 60 * 1000

                        _df_week["week_num"] = ((_df_week["finishedAt"] - _anchor_monday_ms) // _ms_per_week + 1).clip(lower=1).astype(int)

                        _week_counts = (
                            _df_week.groupby("week_num")["result"].count()
                            .sort_index()
                            .head(10)
                        )
                        _week_nums_ordered = list(_week_counts.index)
                        _df_week = _df_week[_df_week["week_num"].isin(_week_nums_ordered)]

                        def _week_label(w):
                            monday = _anchor_monday + timedelta(weeks=int(w) - 1)
                            return f"W{w}  ({monday.day} {monday.strftime('%b')})"

                        _df_week["week"] = _df_week["week_num"].map(_week_label)
                        _week_order = [_week_label(w) for w in _week_nums_ordered]

                        week_overview = _build_period_overview(_df_week, "week", _week_order)
                        st.plotly_chart(
                            _make_overview_chart(week_overview, "week", _week_order, f"{_sel_season} Weekly Overview - {player_label}"),
                            width='stretch',
                        )
                        week_opp_df = _build_period_opp_df(_df_week, "week", _week_order)
                        st.plotly_chart(
                            _make_period_opp_chart(week_opp_df, _week_order, "Week", f"Top {top_n_season} Opponents per Week ({_sel_season}) - {player_label}"),
                            width='stretch',
                        )


                # -- SECTION 3: CHARACTER MATCHUPS --------------------------------
                st.header("Character Matchups")

                df_heatmap = pm_filtered
                season_label_heatmap = season_label


                def make_char_pie(df_col: pd.Series, title: str, opp_strs: list | None = None) -> go.Figure:
                    counts = df_col.dropna().value_counts().reset_index()
                    counts.columns = ["character", "count"]
                    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
                    colors = [char_color_map.get(c, "#AAAAAA") for c in counts["character"]]
                    extra_line = "<br>%{customdata}" if opp_strs is not None else ""
                    fig = go.Figure(go.Pie(
                        labels=counts["character"],
                        values=counts["count"],
                        marker=dict(colors=colors),
                        direction="clockwise",
                        sort=False,
                        textinfo="label+percent",
                        textposition="auto",
                        textfont=dict(size=11),
                        customdata=opp_strs if opp_strs is not None else [None] * len(counts),
                        hovertemplate=f"<b>%{{label}}</b><br>Games: %{{value}}  |  Share: %{{percent}}{extra_line}<extra></extra>",
                    ))
                    fig.update_layout(title=title, height=300, showlegend=False,
                                      hoverlabel=dict(align="left"),
                                      margin=dict(t=40, b=10, l=10, r=10))
                    return fig


                def _top_opps_str(df: pd.DataFrame, filter_col: str, char: str, opp_col: str, top_n: int = 4) -> str:
                    sub = df[df[filter_col] == char][opp_col].dropna().value_counts()
                    total = sub.sum()
                    if total == 0:
                        return "-"
                    tops = [f"{o} {sub[o]/total*100:.0f}%" for o in sub.index[:top_n]]
                    line1 = "  ·  ".join(tops[:2])
                    line2 = "  ·  ".join(tops[2:4]) if len(tops) > 2 else ""
                    return f"{line1}<br>{line2}" if line2 else line1


                col_pie1, col_pie2 = st.columns(2)
                _my_chars = df_heatmap["player_char_name"].dropna().value_counts().sort_values(ascending=False).index.tolist()
                _opp_chars = df_heatmap["opp_char_name"].dropna().value_counts().sort_values(ascending=False).index.tolist()
                _my_opp_strs = [_top_opps_str(df_heatmap, "player_char_name", c, "opp_label") for c in _my_chars]
                _opp_opp_strs = [_top_opps_str(df_heatmap, "opp_char_name", c, "opp_label") for c in _opp_chars]
                with col_pie1:
                    st.plotly_chart(
                        make_char_pie(df_heatmap["player_char_name"], f"Your Character Picks - {season_label_heatmap} ({player_label})", _my_opp_strs),
                        width='stretch',
                    )
                with col_pie2:
                    st.plotly_chart(
                        make_char_pie(df_heatmap["opp_char_name"], f"Opponent Character Picks - {season_label_heatmap}", _opp_opp_strs),
                        width='stretch',
                    )


                if df_heatmap["player_char_name"].notna().any():
                    st.plotly_chart(
                        make_character_matchup_chart(
                            df_heatmap,
                            f"Character Matchup Win Rate - {season_label_heatmap} ({player_label})",
                        ),
                        width='stretch',
                    )
                else:
                    st.info("No character data for this selection.")


# -- TAB: RANKINGS -------------------------------------------------------------
elif _main_view == "Leaderboard":
    _rankings.render(matches, player_label_map, _reset_to_season, _reset_to_ladder_name, char_map, char_color_map)
