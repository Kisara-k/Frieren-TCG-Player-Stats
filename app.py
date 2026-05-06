import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Frieren TCG Player Stats", layout="wide")

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
@st.cache_data
def load_data():
    players = pd.read_csv("data/Player.csv")
    matches = pd.read_csv("data/Match.csv")
    characters = pd.read_csv("data/Character.csv")
    players["name"] = players["name"].replace("", pd.NA).fillna(players["discordName"])
    return players, matches, characters

players, matches, characters = load_data()

player_label_map = {
    row["id"]: (
        row["name"]
        if ("name" in players.columns and pd.notna(row.get("name")) and str(row.get("name", "")).strip() != "")
        else str(row["id"])
    )
    for _, row in players.iterrows()
}
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
    player_label = (
        row["name"].values[0]
        if ("name" in players.columns and pd.notna(row["name"].values[0]) and str(row["name"].values[0]).strip() != "")
        else str(player_id)
    )

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
    pm["season"] = "S" + pm["ladderResetId"].astype(str)
    pm["player_char_name"] = pm["player_char"].map(char_map)
    pm["opp_char_name"] = pm["opp_char"].map(char_map)
    pm["opp_label"] = pm["opp_id"].map(player_label_map)
    # Normalise ranked: null → 0 (unranked), any non-null value → 1 (ranked)
    pm["ranked"] = pm["ranked"].notna().astype(int)

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


# -- PLAYER SEARCH -------------------------------------------------------------
st.title("Frieren TCG Player Stats")

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
        on_change=_commit_by_name,   # fires immediately when a name is chosen
    )
with col_id:
    st.text_input(
        "Discord ID",
        placeholder="Or paste Discord ID…",
        label_visibility="collapsed",
        key="discord_id_input",
        on_change=_commit_by_id,     # fires on Enter or blur in the text box
    )
with col_btn:
    # Fallback button for mouse-only users
    if st.button("Analyze", type="primary", use_container_width=True):
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
    st.stop()

# -- LOAD / CACHE PLAYER DATA --------------------------------------------------
try:
    pm, player_label, season_counts = build_player_matches(confirmed_id)
except (ValueError, OverflowError):
    st.error("Invalid Discord ID - must be a numeric snowflake.")
    st.stop()

if pm is None or pm.empty:
    st.error(f"No player found with Discord ID `{confirmed_id}`.")
    st.stop()

st.subheader(f"{player_label}")

# -- RANKED FILTER ------------------------------------------------------------
ranked_mode = st.radio(
    "Match type",
    options=["All", "Ranked", "Unranked"],
    horizontal=True,
    key="ranked_filter",
    label_visibility="collapsed",
)

if ranked_mode == "Ranked":
    pm_filtered = pm[pm["ranked"] == 1].copy()
elif ranked_mode == "Unranked":
    pm_filtered = pm[pm["ranked"] == 0].copy()
else:
    pm_filtered = pm

# Recompute season counts from the filtered set for accurate dropdown labels
_filtered_season_counts = (
    pm_filtered.groupby("season")["result"]
    .count()
    .reindex(sorted(pm_filtered["season"].unique(), key=lambda s: int(s[1:])) if not pm_filtered.empty else [])
    .dropna()
    .astype(int)
    .to_dict()
) if not pm_filtered.empty else {}

seasons = list(_filtered_season_counts.keys())
total_games = sum(_filtered_season_counts.values())

_season_breakdown = "  ·  ".join(f"{s}: {n}" for s, n in _filtered_season_counts.items())
st.caption(f"Total matches: {total_games}  |  {_season_breakdown}")

# Dropdown labels built from the filtered season counts
all_option = f"All  ({total_games} games)"
season_options = [all_option] + [f"{s}  ({n} games)" for s, n in _filtered_season_counts.items()]

def filter_by_season_option(option: str) -> pd.DataFrame:
    if option == all_option:
        return pm_filtered
    season_key = option.split("  ")[0]
    return pm_filtered[pm_filtered["season"] == season_key]

def season_display_label(option: str) -> str:
    return "All Seasons" if option == all_option else option.split("  ")[0]


# -- SECTION 1: OVERALL MATCHUPS ------------------------------------------------
st.header("Overall Matchups")

season_sel_matchup = st.selectbox(
    "Season filter", season_options, key="matchup_season",
    label_visibility="collapsed",
)
df_matchup = filter_by_season_option(season_sel_matchup)
season_label_matchup = season_display_label(season_sel_matchup)

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
long20 = long20.merge(top20[["opp_label", "Games", "WinRate", "MyPicks", "OppPicks"]], on="opp_label")
fig1 = px.bar(
    long20, x="opp_label", y="Count", color="Result",
    color_discrete_map={"Wins": "#2ecc71", "Losses": "#e74c3c"},
    title=f"Top 20 Opponents by Games Played - {season_label_matchup} ({player_label})",
    labels={"opp_label": "Opponent", "Count": "Games"},
    text_auto=True, barmode="stack",
    category_orders={"opp_label": top20["opp_label"].tolist()},
    custom_data=["Games", "WinRate", "MyPicks", "OppPicks"],
)
fig1.update_traces(
    hovertemplate=(
        "<b>%{x}</b><br>"
        "%{fullData.name}: %{y}  |  Total: %{customdata[0]}  |  WR: %{customdata[1]}%<br>"
        "<b>Your picks:</b> %{customdata[2]}<br>"
        "<b>Their picks:</b> %{customdata[3]}"
        "<extra></extra>"
    )
)
fig1.update_layout(legend_title_text="Result")
st.plotly_chart(fig1, use_container_width=True)

min_games = st.slider("Minimum games for win-rate chart", 1, 10, 5, key="min_games_slider")
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
st.plotly_chart(fig2, use_container_width=True)


# -- SECTION 2: PER-SEASON BREAKDOWN -------------------------------------------
st.header("Per-Season Breakdown")

season_overview = pm_filtered.groupby("season").agg(
    Games=("result", "count"),
    Wins=("result", lambda x: (x == "Win").sum()),
    Losses=("result", lambda x: (x == "Loss").sum()),
).reset_index()
season_overview = season_overview.iloc[
    season_overview["season"].str[1:].astype(int).argsort()
].reset_index(drop=True)
season_overview["WinRate"] = (season_overview["Wins"] / season_overview["Games"] * 100).round(1)
season_overview["MyPicks"] = season_overview["season"].map(
    lambda s: char_picks_str(pm_filtered[pm_filtered["season"] == s]["player_char_name"])
)
season_overview["OppPicks"] = season_overview["season"].map(
    lambda s: char_picks_str(pm_filtered[pm_filtered["season"] == s]["opp_char_name"])
)
_sov_cd = season_overview[["WinRate", "MyPicks", "OppPicks"]].values

fig_sov = make_subplots(specs=[[{"secondary_y": True}]])
fig_sov.add_trace(go.Bar(
    x=season_overview["season"], y=season_overview["Wins"],
    name="Wins", marker_color="#2ecc71",
    text=season_overview["Wins"], textposition="inside",
    customdata=_sov_cd,
    hovertemplate=(
        "<b>%{x}</b><br>Wins: %{y}  |  WR: %{customdata[0]}%<br>"
        "<b>Your picks:</b> %{customdata[1]}<br>"
        "<b>Their picks:</b> %{customdata[2]}<extra></extra>"
    ),
), secondary_y=False)
fig_sov.add_trace(go.Bar(
    x=season_overview["season"], y=season_overview["Losses"],
    name="Losses", marker_color="#e74c3c",
    text=season_overview["Losses"], textposition="inside",
    customdata=_sov_cd,
    hovertemplate=(
        "<b>%{x}</b><br>Losses: %{y}  |  WR: %{customdata[0]}%<br>"
        "<b>Your picks:</b> %{customdata[1]}<br>"
        "<b>Their picks:</b> %{customdata[2]}<extra></extra>"
    ),
), secondary_y=False)
fig_sov.add_trace(go.Scatter(
    x=season_overview["season"], y=season_overview["WinRate"],
    name="Win Rate %", mode="lines+markers+text",
    line=dict(color="#3498db", width=2), marker=dict(size=8),
    text=season_overview["WinRate"].astype(str) + "%", textposition="top center",
), secondary_y=True)
fig_sov.update_layout(
    title=f"Season Overview - {player_label}",
    barmode="stack", legend_title_text="",
)
fig_sov.update_xaxes(categoryorder="array", categoryarray=season_overview["season"].tolist())
fig_sov.update_yaxes(title_text="Games", secondary_y=False)
fig_sov.update_yaxes(title_text="Win Rate (%)", range=[0, 110], secondary_y=True)
st.plotly_chart(fig_sov, use_container_width=True)

# Per-season top opponents
top_n_season = 6
rows = []
for season in seasons:
    df_s = pm_filtered[pm_filtered["season"] == season]
    stats_s = matchup_stats(df_s, top_n=top_n_season)
    for rank, (_, row) in enumerate(stats_s.iterrows(), start=1):
        rows.append({
            "season": season,
            "opp_label": row["opp_label"],
            "rank": rank,
            "Wins": int(row["Wins"]),
            "Losses": int(row["Losses"]),
            "Games": int(row["Games"]),
            "WinRate": row["WinRate"],
        })

season_opp_df = pd.DataFrame(rows)
_sopp_picks = (
    pm_filtered.groupby(["season", "opp_label"])[["player_char_name", "opp_char_name"]]
    .apply(lambda g: pd.Series({
        "MyPicks": char_picks_str(g["player_char_name"]),
        "OppPicks": char_picks_str(g["opp_char_name"]),
    }))
    .reset_index()
)
season_opp_df = season_opp_df.merge(_sopp_picks, on=["season", "opp_label"], how="left")
mid = len(seasons) // 2
season_splits = [seasons[:mid], seasons[mid:]]
split_labels = ["(Early Seasons)", "(Late Seasons)"]


def make_season_opp_chart(season_subset, subtitle):
    df_sub = season_opp_df[season_opp_df["season"].isin(season_subset)].copy()
    fig = go.Figure()
    wins_in_legend = False
    losses_in_legend = False

    for rank in range(1, top_n_season + 1):
        df_rank = df_sub[df_sub["rank"] == rank].copy()
        if df_rank.empty:
            continue
        og = f"rank{rank}"
        _cd_w = df_rank[["opp_label", "Games", "WinRate", "Losses", "MyPicks", "OppPicks"]].values
        _cd_l = df_rank[["opp_label", "Games", "WinRate", "Wins", "MyPicks", "OppPicks"]].values
        fig.add_trace(go.Bar(
            name="Wins", x=df_rank["season"], y=df_rank["Wins"],
            marker_color="#2ecc71", legendgroup="Wins",
            showlegend=not wins_in_legend, offsetgroup=og,
            customdata=_cd_w,
            hovertemplate=(
                "<b>%{x}</b> - Rank #" + str(rank) + "<br>"
                "%{customdata[0]}<br>"
                "Wins: %{y}  Losses: %{customdata[3]}<br>"
                "Games: %{customdata[1]}  WR: %{customdata[2]}%<br>"
                "<b>Your picks:</b> %{customdata[4]}<br>"
                "<b>Their picks:</b> %{customdata[5]}"
                "<extra></extra>"
            ),
        ))
        wins_in_legend = True
        fig.add_trace(go.Bar(
            name="Losses", x=df_rank["season"], y=df_rank["Losses"],
            base=df_rank["Wins"].tolist(),
            marker_color="#e74c3c", legendgroup="Losses",
            showlegend=not losses_in_legend, offsetgroup=og,
            text=df_rank["opp_label"], textposition="outside",
            textangle=-90, textfont=dict(size=11),
            outsidetextfont=dict(size=11), constraintext="none", cliponaxis=False,
            customdata=_cd_l,
            hovertemplate=(
                "<b>%{x}</b> - Rank #" + str(rank) + "<br>"
                "%{customdata[0]}<br>"
                "Wins: %{customdata[3]}  Losses: %{y}<br>"
                "Games: %{customdata[1]}  WR: %{customdata[2]}%<br>"
                "<b>Your picks:</b> %{customdata[4]}<br>"
                "<b>Their picks:</b> %{customdata[5]}"
                "<extra></extra>"
            ),
        ))
        losses_in_legend = True

    max_label_len = df_sub["opp_label"].str.len().max() if not df_sub.empty else 10
    label_top_margin = max(120, max_label_len * 8)
    fig.update_layout(
        title=f"Top {top_n_season} Opponents per Season - {player_label} {subtitle}",
        barmode="group",
        xaxis=dict(title="Season", categoryorder="array", categoryarray=season_subset),
        yaxis_title="Games",
        legend_title_text="",
        height=420 + label_top_margin,
        margin=dict(t=label_top_margin, b=60, l=60, r=20),
        bargap=0.15, bargroupgap=0.05,
    )
    return fig


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        make_season_opp_chart(season_splits[0], split_labels[0]),
        use_container_width=True,
    )
with col2:
    st.plotly_chart(
        make_season_opp_chart(season_splits[1], split_labels[1]),
        use_container_width=True,
    )


# -- SECTION 3: CHARACTER MATCHUPS --------------------------------------------
st.header("Character Matchups")

season_sel_heatmap = st.selectbox(
    "Season filter", season_options, key="heatmap_season",
    label_visibility="collapsed",
)
df_heatmap = filter_by_season_option(season_sel_heatmap)
season_label_heatmap = season_display_label(season_sel_heatmap)


def make_char_pie(df_col: pd.Series, title: str) -> go.Figure:
    counts = df_col.dropna().value_counts().reset_index()
    counts.columns = ["character", "count"]
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
    colors = [char_color_map.get(c, "#AAAAAA") for c in counts["character"]]
    fig = go.Figure(go.Pie(
        labels=counts["character"],
        values=counts["count"],
        marker=dict(colors=colors),
        direction="clockwise",
        sort=False,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Games: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(title=title, height=300, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    return fig


col_pie1, col_pie2 = st.columns(2)
with col_pie1:
    st.plotly_chart(
        make_char_pie(df_heatmap["player_char_name"], f"Your Character Picks - {season_label_heatmap}"),
        use_container_width=True,
    )
with col_pie2:
    st.plotly_chart(
        make_char_pie(df_heatmap["opp_char_name"], f"Opponent Character Picks - {season_label_heatmap}"),
        use_container_width=True,
    )


def char_heatmap(df, title):
    agg = df.groupby(["player_char_name", "opp_char_name"]).agg(
        Games=("result", "count"),
        Wins=("result", lambda x: (x == "Win").sum()),
    ).reset_index()
    agg["WinRate"] = (agg["Wins"] / agg["Games"] * 100).round(1)
    agg["Losses"] = agg["Games"] - agg["Wins"]

    wr_matrix = agg.pivot(index="player_char_name", columns="opp_char_name", values="WinRate")
    g_matrix  = agg.pivot(index="player_char_name", columns="opp_char_name", values="Games")
    w_matrix  = agg.pivot(index="player_char_name", columns="opp_char_name", values="Wins").fillna(0)
    l_matrix  = agg.pivot(index="player_char_name", columns="opp_char_name", values="Losses").fillna(0)

    row_wins   = w_matrix.sum(axis=1).astype(int)
    row_losses = l_matrix.sum(axis=1).astype(int)
    col_wins   = w_matrix.sum(axis=0).astype(int)
    col_losses = l_matrix.sum(axis=0).astype(int)

    text_vals = []
    for r in wr_matrix.index:
        row_text = []
        for c in wr_matrix.columns:
            wr = wr_matrix.loc[r, c]
            g  = g_matrix.loc[r, c] if r in g_matrix.index and c in g_matrix.columns else np.nan
            row_text.append(f"{wr:.0f}%<br>({int(g)})" if pd.notna(wr) else "")
        text_vals.append(row_text)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.15, 0.85],
        row_heights=[0.85, 0.15],
        shared_xaxes="columns",
        shared_yaxes="rows",
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    fig.add_trace(go.Heatmap(
        z=wr_matrix.values,
        x=list(wr_matrix.columns),
        y=list(wr_matrix.index),
        text=text_vals,
        texttemplate="%{text}",
        colorscale="Blues",
        zmin=0, zmax=100,
        colorbar=dict(title="Win %"),
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=row_wins[wr_matrix.index].values,
        y=list(wr_matrix.index),
        orientation="h", marker_color="#2ecc71",
        name="Wins", legendgroup="Wins", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=row_losses[wr_matrix.index].values,
        y=list(wr_matrix.index),
        orientation="h", marker_color="#e74c3c",
        name="Losses", legendgroup="Losses", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=list(wr_matrix.columns),
        y=col_wins[wr_matrix.columns].values,
        marker_color="#2ecc71", name="Wins", legendgroup="Wins", showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=list(wr_matrix.columns),
        y=col_losses[wr_matrix.columns].values,
        marker_color="#e74c3c", name="Losses", legendgroup="Losses", showlegend=False,
    ), row=2, col=2)

    fig.update_xaxes(autorange="reversed", showticklabels=True, row=1, col=1)
    fig.update_yaxes(title_text="Player's Character", row=1, col=1)
    fig.update_xaxes(title_text="Opponent's Character", row=2, col=2)
    fig.update_yaxes(showticklabels=True, row=2, col=2)
    fig.update_layout(
        title=title,
        barmode="stack",
        height=max(420, len(wr_matrix.index) * 55 + 200),
    )
    return fig


if df_heatmap["player_char_name"].notna().any():
    st.plotly_chart(
        char_heatmap(df_heatmap, f"Character Matchup Win Rate - {season_label_heatmap} ({player_label})"),
        use_container_width=True,
    )
else:
    st.info("No character data for this selection.")
