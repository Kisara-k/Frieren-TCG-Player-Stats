import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from html import escape
from matchup_chart import make_character_matchup_chart

_LADDER_OPTIONS = ["Classic", "All", "Blitz", "Slow", "Prescience"]
_LADDER_RAW = {
    "Classic": "classic", "Blitz": "blitz",
    "Slow": "slow", "Prescience": "classic-prescience", "All": None,
}


def _leaderboard(m, player_label_map):
    w = (
        m.groupby("winnerId")
        .agg(wins=("id", "count"), ranked_wins=("ranked_flag", "sum"))
        .rename_axis("player_id")
    )
    l = (
        m.groupby("loserId")
        .agg(losses=("id", "count"), ranked_losses=("ranked_flag", "sum"))
        .rename_axis("player_id")
    )
    s = w.join(l, how="outer").fillna(0).astype(int)
    s["total_games"] = s["wins"] + s["losses"]
    s["ranked_games"] = s["ranked_wins"] + s["ranked_losses"]
    s["unranked_games"] = s["total_games"] - s["ranked_games"]
    s["win_rate"] = (s["wins"] / s["total_games"] * 100).round(1)
    s["player_name"] = s.index.map(player_label_map)
    return s.sort_values("total_games", ascending=False).reset_index()


def _top_players_str(m, char_id, player_label_map, top_n=4):
    winners = m[m["winnerCharacterId"] == char_id]["winnerId"]
    losers = m[m["loserCharacterId"] == char_id]["loserId"]
    sub = pd.concat([winners, losers]).value_counts()
    total = sub.sum()
    if total == 0:
        return "-"
    tops = [f"{player_label_map.get(pid, str(pid))} {sub[pid]/total*100:.0f}%" for pid in sub.index[:top_n]]
    line1 = "  ·  ".join(tops[:2])
    line2 = "  ·  ".join(tops[2:4]) if len(tops) > 2 else ""
    return f"{line1}<br>{line2}" if line2 else line1


def _hex_to_hsl_capped(hex_color, max_l=60):
    """Keep character colors readable against Plotly's light hover background."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin
    lightness = (cmax + cmin) / 2
    saturation = 0.0 if delta == 0 else delta / (1 - abs(2 * lightness - 1))
    if delta == 0:
        hue = 0.0
    elif cmax == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        hue = 60 * ((b - r) / delta + 2)
    else:
        hue = 60 * ((r - g) / delta + 4)
    capped = min(lightness, max_l / 100)
    if lightness > capped:
        saturation *= capped / lightness
    return f"hsl({hue:.0f},{saturation * 100:.0f}%,{capped * 100:.0f}%)"


def _char_picks_str(character_ids, char_map, char_color_map, top_n=4):
    counts = character_ids.dropna().map(char_map).dropna().value_counts()
    total = int(counts.sum())
    if not total:
        return "-"
    return "  ·  ".join(
        f'<span style="color:{_hex_to_hsl_capped(char_color_map.get(name, "#AAAAAA"))}">'
        f'<b>{escape(str(name))}</b> {count / total * 100:.0f}%</span>'
        for name, count in counts.iloc[:top_n].items()
    )


def _player_activity_tooltip(m, player_id, player_label_map, char_map, char_color_map):
    """Build a player's pick summary and five most-played opponent lines."""
    wins = m[m["winnerId"] == player_id][
        ["winnerCharacterId", "loserId", "loserCharacterId"]
    ].rename(columns={
        "winnerCharacterId": "player_char", "loserId": "opp_id",
        "loserCharacterId": "opp_char",
    })
    losses = m[m["loserId"] == player_id][
        ["loserCharacterId", "winnerId", "winnerCharacterId"]
    ].rename(columns={
        "loserCharacterId": "player_char", "winnerId": "opp_id",
        "winnerCharacterId": "opp_char",
    })
    perspective = pd.concat([wins, losses], ignore_index=True)
    player_picks = _char_picks_str(
        perspective["player_char"], char_map, char_color_map
    )
    lines = [f"<b>Top picks:</b> {player_picks}", "<b>Top opponents:</b>"]
    for opp_id, games in perspective["opp_id"].value_counts().iloc[:5].items():
        opp_name = escape(str(player_label_map.get(opp_id, opp_id)))
        opp_picks = _char_picks_str(
            perspective.loc[perspective["opp_id"] == opp_id, "opp_char"],
            char_map, char_color_map,
        )
        game_label = "game" if games == 1 else "games"
        lines.append(f"{int(games)} {game_label} · <b>{opp_name}</b> · {opp_picks}")
    return "<br>".join(lines)


def _char_meta(m, char_map, char_color_map):
    char_wins = m.groupby("winnerCharacterId").size().rename("wins")
    char_losses = m.groupby("loserCharacterId").size().rename("losses")
    cs = pd.concat([char_wins, char_losses], axis=1).fillna(0).astype(int)
    cs.index.name = "char_id"
    cs["total"] = cs["wins"] + cs["losses"]
    cs["win_rate"] = (cs["wins"] / cs["total"] * 100).round(1)
    cs["name"] = cs.index.map(char_map)
    cs["color"] = cs["name"].map(lambda n: char_color_map.get(n, "#AAAAAA"))
    return cs.dropna(subset=["name"]).sort_values("total", ascending=False).reset_index()


def render(matches, player_label_map, reset_to_season, reset_to_ladder_name, char_map, char_color_map):
    # Base: exclude self-matches, add derived columns
    m_base = matches[matches["winnerId"] != matches["loserId"]].copy()
    m_base["season"] = m_base["ladderResetId"].map(reset_to_season)
    m_base["ladder_name"] = m_base["ladderResetId"].map(reset_to_ladder_name)
    m_base["ranked_flag"] = m_base["ranked"].notna().astype(int)

    all_seasons = sorted(m_base["season"].dropna().unique(), key=lambda s: int(s[1:]))
    all_seasons_desc = list(reversed(all_seasons))

    # Build season counts in the current ladder/match-type context. Streamlit
    # reruns after either radio changes, so these labels stay in sync.
    count_matches = m_base
    count_ladder = _LADDER_RAW.get(st.session_state.get("rnk_ladder", "Classic"))
    if count_ladder:
        count_matches = count_matches[count_matches["ladder_name"] == count_ladder]
    count_ranked_mode = st.session_state.get("rnk_ranked", "All")
    if count_ranked_mode == "Ranked":
        count_matches = count_matches[count_matches["ranked_flag"] == 1]
    elif count_ranked_mode == "Unranked":
        count_matches = count_matches[count_matches["ranked_flag"] == 0]
    season_game_counts = count_matches["season"].value_counts().to_dict()

    # --- Filters ---
    col_s, col_l, col_r = st.columns([1.5, 2.5, 2])
    _ALL_OPT = "All Seasons"

    def _season_option_label(season):
        games = len(count_matches) if season == _ALL_OPT else season_game_counts.get(season, 0)
        return f"{season} ({games:,} games)"

    if "rnk_seasons" not in st.session_state:
        st.session_state["rnk_seasons"] = all_seasons_desc[:1] or [_ALL_OPT]
    if "rnk_seasons_previous" not in st.session_state:
        st.session_state["rnk_seasons_previous"] = list(
            st.session_state["rnk_seasons"]
        )

    def _sync_leaderboard_seasons():
        current = list(st.session_state.get("rnk_seasons", []))
        previous = list(st.session_state.get("rnk_seasons_previous", []))
        if _ALL_OPT in current and _ALL_OPT not in previous:
            current = [_ALL_OPT]
        elif _ALL_OPT in current and any(s != _ALL_OPT for s in current):
            current = [s for s in current if s != _ALL_OPT]
        st.session_state["rnk_seasons"] = current
        st.session_state["rnk_seasons_previous"] = current

    with col_s:
        season_sel = st.multiselect(
            "Season", [_ALL_OPT] + all_seasons_desc,
            key="rnk_seasons",
            format_func=_season_option_label,
            on_change=_sync_leaderboard_seasons,
            placeholder="Select seasons", label_visibility="collapsed",
        )
    with col_l:
        ladder_mode = st.radio(
            "Ladder", _LADDER_OPTIONS, index=0, horizontal=True,
            key="rnk_ladder", label_visibility="collapsed",
        )
    with col_r:
        ranked_mode = st.radio(
            "Match type", ["All", "Ranked", "Unranked"], horizontal=True,
            key="rnk_ranked", label_visibility="collapsed",
        )

    # Apply ladder + ranked filters (season applied separately below)
    ladder_raw = _LADDER_RAW[ladder_mode]
    m_filtered = m_base[m_base["ladder_name"] == ladder_raw].copy() if ladder_raw else m_base.copy()
    if ranked_mode == "Ranked":
        m_filtered = m_filtered[m_filtered["ranked_flag"] == 1]
    elif ranked_mode == "Unranked":
        m_filtered = m_filtered[m_filtered["ranked_flag"] == 0]

    if not season_sel or _ALL_OPT in season_sel:
        m_view = m_filtered
        season_label = "All Seasons"
        selected_seasons = all_seasons
    else:
        selected_seasons = [s for s in all_seasons if s in season_sel]
        m_view = m_filtered[m_filtered["season"].isin(selected_seasons)]
        season_label = ", ".join(selected_seasons)

    if m_view.empty:
        st.info("No matches for this selection.")
        return

    lb = _leaderboard(m_view, player_label_map)
    total_matches = len(m_view)
    ranked_matches = int(m_view["ranked_flag"].sum())

    # ── Summary metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Players", len(lb))
    c2.metric("Total Matches", f"{total_matches:,}")
    c3.metric("Ranked Matches", f"{ranked_matches:,}")
    c4.metric("Ranked %", f"{ranked_matches / total_matches * 100:.0f}%" if total_matches else "-")

    max_g = int(lb["total_games"].max()) if not lb.empty else 1
    _default_min = 1
    for _t in range(1, max_g + 1):
        if len(lb[lb["total_games"] >= _t]) <= 30:
            _default_min = _t
            break

    # Reset slider when the filter context (season/ladder/ranked) changes so
    # the stale value from the previous context isn't used.
    _ctx = (tuple(season_sel), ladder_mode, ranked_mode)
    if st.session_state.get("rnk_filter_ctx") != _ctx:
        st.session_state["rnk_min_games"] = _default_min
        st.session_state["rnk_char_top_n"] = min(20, len(lb))
        st.session_state["rnk_filter_ctx"] = _ctx
    else:
        # Streamlit removes widget-owned keys when their view is not rendered.
        # Restore their computed defaults when returning to the leaderboard.
        if "rnk_min_games" not in st.session_state:
            st.session_state["rnk_min_games"] = _default_min
        if "rnk_char_top_n" not in st.session_state:
            st.session_state["rnk_char_top_n"] = min(20, len(lb))

    # Read slider value from session state so the bubble chart and table
    # can use it before the slider widget is rendered below.
    min_g = int(st.session_state.get("rnk_min_games", _default_min))
    min_g = max(1, min(min_g, max_g))
    st.session_state["rnk_min_games"] = min_g

    st.subheader(f"Player Rankings")

    # ── Activity bar chart (top 20) ────────────────────────────────────────────
    top20 = lb.head(20).copy()
    top20["matchup_tooltip"] = top20["player_id"].map(
        lambda player_id: _player_activity_tooltip(
            m_view, player_id, player_label_map, char_map, char_color_map
        )
    )
    activity_customdata = top20[
        ["total_games", "wins", "losses", "win_rate", "matchup_tooltip"]
    ].values
    fig_act = go.Figure([
        go.Bar(
            name="Wins", x=top20["player_name"], y=top20["wins"],
            marker_color="#2ecc71", text=top20["wins"], textposition="inside",
            customdata=activity_customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Wins %{customdata[1]}  |  Losses %{customdata[2]}  |  "
                "Total %{customdata[0]}  |  WR: %{customdata[3]}%<br>"
                "%{customdata[4]}"
                "<extra></extra>"
            ),
        ),
        go.Bar(
            name="Losses", x=top20["player_name"], y=top20["losses"],
            marker_color="#e74c3c", text=top20["losses"], textposition="inside",
            customdata=activity_customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Wins %{customdata[1]}  |  Losses %{customdata[2]}  |  "
                "Total %{customdata[0]}  |  WR: %{customdata[3]}%<br>"
                "%{customdata[4]}"
                "<extra></extra>"
            ),
        ),
    ])
    fig_act.update_layout(
        title=f"Top 20 Most Active Players - {season_label}",
        barmode="stack", xaxis_title="Player", yaxis_title="Games",
        xaxis={"categoryorder": "array", "categoryarray": top20["player_name"].tolist()},
        legend_title_text="Result", hoverlabel=dict(align="left"),
    )
    st.plotly_chart(fig_act, width='stretch')

    # ── Win rate bubble chart ──────────────────────────────────────────────────
    lb_wr = lb[lb["total_games"] >= min_g].copy()
    fig_wr = px.scatter(
        lb_wr, x="total_games", y="win_rate",
        size="total_games", color="win_rate",
        color_continuous_scale="RdYlGn", range_color=[0, 100],
        text="player_name",
        hover_data={"player_name": True, "wins": True, "losses": True, "total_games": True, "win_rate": True},
        labels={"total_games": "Games Played", "win_rate": "Win Rate (%)", "player_name": "Player"},
        title=f"Win Rate vs Activity - {season_label} (Top {len(lb_wr)})",
    )
    fig_wr.update_traces(textposition="top center", textfont_size=9)
    fig_wr.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig_wr.update_layout(coloraxis_showscale=False, height=520, hoverlabel=dict(align="left"))
    st.plotly_chart(fig_wr, width='stretch')

    st.slider("Minimum games", 1, max(2, max_g), key="rnk_min_games")

    # ── Player leaderboard ─────────────────────────────────────────────────────
    lb_show = lb[lb["total_games"] >= min_g].copy()
    st.markdown(f"<p style='font-size:17px;font-weight:500;margin:0 0 4px 0;'>Player Leaderboard - {season_label} (Top {len(lb_show)})</p>", unsafe_allow_html=True)
    lb_show.insert(0, "Rank", range(1, len(lb_show) + 1))
    display = lb_show[["Rank", "player_name", "total_games", "wins", "losses", "win_rate", "ranked_games", "unranked_games"]].copy()
    display.columns = ["Rank", "Player", "Games", "Wins", "Losses", "Win Rate %", "Ranked", "Unranked"]
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Win Rate %": st.column_config.ProgressColumn(
                "Win Rate %", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )

    # ── Season activity heatmap (multi-season selections) ─────────────────────
    if len(selected_seasons) > 1:
        st.subheader("Season Activity - Top 20 Players")
        top_pids = set(lb.head(20)["player_id"])
        w_s = m_filtered[m_filtered["winnerId"].isin(top_pids)][["winnerId", "season"]].rename(columns={"winnerId": "player_id"})
        l_s = m_filtered[m_filtered["loserId"].isin(top_pids)][["loserId", "season"]].rename(columns={"loserId": "player_id"})
        both = pd.concat([w_s, l_s])
        both["player_name"] = both["player_id"].map(player_label_map)
        hm_pivot = (
            both.groupby(["player_name", "season"]).size()
            .unstack(fill_value=0)
            .reindex(columns=selected_seasons, fill_value=0)
        )
        hm_pivot = hm_pivot.loc[hm_pivot.sum(axis=1).sort_values(ascending=False).index]
        fig_hm = go.Figure(go.Heatmap(
            z=hm_pivot.values,
            x=list(hm_pivot.columns),
            y=list(hm_pivot.index),
            colorscale="Blues",
            text=hm_pivot.values.astype(int),
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> - %{x}<br>Games: %{z}<extra></extra>",
            colorbar=dict(title="Games"),
        ))
        fig_hm.update_layout(
            title="Games Played per Player per Season",
            xaxis_title="Season", yaxis_title="Player",
            height=max(400, len(hm_pivot) * 35 + 150),
            hoverlabel=dict(align="left"),
        )
        st.plotly_chart(fig_hm, width='stretch')

    # ── Character meta ─────────────────────────────────────────────────────────
    st.subheader(f"Character Stats")
    max_top_n = len(lb)
    if max_top_n > 2:
        top_n_players = st.slider(
            "Top N players", 2, max_top_n,
            key="rnk_char_top_n",
            help="Only matches where both players are in the top N by games played are included.",
        )
    else:
        top_n_players = max_top_n
    top_player_ids = set(lb.head(top_n_players)["player_id"])
    m_char = m_view[
        m_view["winnerId"].isin(top_player_ids)
        & m_view["loserId"].isin(top_player_ids)
    ].copy()
    # st.caption(
    #     f"Character stats include {len(m_char):,} games where both players are in "
    #     f"the top {top_n_players} by games played."
    # )

    cs = _char_meta(m_char, char_map, char_color_map)
    cs["top_players"] = [_top_players_str(m_char, cid, player_label_map) for cid in cs["char_id"]]

    if cs.empty:
        st.info("No character data for this selection.")
        return

    col_a, col_b = st.columns(2)

    # Pick share
    with col_a:
        fig_picks = go.Figure(go.Bar(
            x=cs["total"], y=cs["name"], orientation="h",
            marker_color=cs["color"],
            customdata=cs[["wins", "losses", "win_rate", "top_players"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Picks: %{x}<br>"
                "Wins %{customdata[0]}  |  Losses %{customdata[1]}  |  WR: %{customdata[2]}%<br>"
                "%{customdata[3]}"
                "<extra></extra>"
            ),
        ))
        fig_picks.update_layout(
            title=f"Character Picked - {season_label}", xaxis_title="Games", yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
            height=max(320, len(cs) * 30 + 100),
            hoverlabel=dict(align="left"),
        )
        st.plotly_chart(fig_picks, width='stretch')

    # Win rate by character
    with col_b:
        cs_wr = cs.sort_values("win_rate", ascending=True)
        fig_wr_char = go.Figure(go.Bar(
            x=cs_wr["win_rate"], y=cs_wr["name"], orientation="h",
            marker_color=cs_wr["color"],
            customdata=cs_wr[["wins", "losses", "total", "top_players"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Win Rate: %{x}%<br>"
                "Wins %{customdata[0]}  |  Losses %{customdata[1]}  |  Total %{customdata[2]}<br>"
                "%{customdata[3]}"
                "<extra></extra>"
            ),
            text=cs_wr["win_rate"].astype(str) + "%",
            textposition="outside",
        ))
        fig_wr_char.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
        fig_wr_char.update_xaxes(range=[0, 110])
        fig_wr_char.update_layout(
            title=f"Win Rate by Character - {season_label}", xaxis_title="Win Rate (%)", yaxis_title="",
            height=max(320, len(cs_wr) * 30 + 100),
            hoverlabel=dict(align="left"),
        )
        st.plotly_chart(fig_wr_char, width='stretch')

    # Character matchup win rate matrix. Add both players' perspectives so the
    # shared chart can aggregate wins/losses identically to the player view.
    matchup_wins = m_char[["winnerCharacterId", "loserCharacterId"]].rename(columns={
        "winnerCharacterId": "player_char_id", "loserCharacterId": "opp_char_id",
    })
    matchup_wins["result"] = "Win"
    matchup_losses = m_char[["loserCharacterId", "winnerCharacterId"]].rename(columns={
        "loserCharacterId": "player_char_id", "winnerCharacterId": "opp_char_id",
    })
    matchup_losses["result"] = "Loss"
    matchup_data = pd.concat([matchup_wins, matchup_losses], ignore_index=True)
    matchup_data["player_char_name"] = matchup_data["player_char_id"].map(char_map)
    matchup_data["opp_char_name"] = matchup_data["opp_char_id"].map(char_map)
    matchup_data = matchup_data.dropna(subset=["player_char_name", "opp_char_name"])
    if not matchup_data.empty:
        fig_mu = make_character_matchup_chart(
            matchup_data, f"Character Matchup Win Rates - {season_label}",
        )
        st.plotly_chart(fig_mu, width='stretch')
