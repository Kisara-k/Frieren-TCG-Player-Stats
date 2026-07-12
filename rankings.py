import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

    # --- Filters ---
    col_s, col_l, col_r = st.columns([1.5, 2.5, 2])
    _ALL_OPT = "All Seasons"
    with col_s:
        season_sel = st.selectbox(
            "Season", [_ALL_OPT] + all_seasons_desc,
            index=1 if all_seasons_desc else 0, key="rnk_season", label_visibility="collapsed",
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

    if season_sel == _ALL_OPT:
        m_view = m_filtered
        season_label = "All Seasons"
    else:
        m_view = m_filtered[m_filtered["season"] == season_sel]
        season_label = season_sel

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
    _ctx = (season_sel, ladder_mode, ranked_mode)
    if st.session_state.get("rnk_filter_ctx") != _ctx:
        st.session_state["rnk_min_games"] = _default_min
        st.session_state["rnk_filter_ctx"] = _ctx

    # Read slider value from session state so the bubble chart and table
    # can use it before the slider widget is rendered below.
    min_g = int(st.session_state.get("rnk_min_games", _default_min))
    min_g = max(1, min(min_g, max_g))

    # ── Activity bar chart (top 20) ────────────────────────────────────────────
    top20 = lb.head(20).copy()
    fig_act = go.Figure([
        go.Bar(
            name="Ranked", x=top20["player_name"], y=top20["ranked_games"],
            marker_color="#3498db",
            customdata=top20[["total_games", "wins", "losses", "win_rate"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Ranked: %{y}  |  Total: %{customdata[0]}<br>"
                "Wins %{customdata[1]}  |  Losses %{customdata[2]}  |  WR: %{customdata[3]}%"
                "<extra></extra>"
            ),
        ),
        go.Bar(
            name="Unranked", x=top20["player_name"], y=top20["unranked_games"],
            marker_color="#95a5a6",
            customdata=top20[["total_games", "wins", "losses", "win_rate"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Unranked: %{y}  |  Total: %{customdata[0]}<br>"
                "Wins %{customdata[1]}  |  Losses %{customdata[2]}  |  WR: %{customdata[3]}%"
                "<extra></extra>"
            ),
        ),
    ])
    fig_act.update_layout(
        title=f"Top 20 Most Active Players - {season_label}",
        barmode="stack", xaxis_title="Player", yaxis_title="Games",
        xaxis={"categoryorder": "array", "categoryarray": top20["player_name"].tolist()},
        legend_title_text="Match Type", hoverlabel=dict(align="left"),
    )
    st.plotly_chart(fig_act, use_container_width=True)

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
    st.plotly_chart(fig_wr, use_container_width=True)

    st.slider("Minimum games", 1, max(2, max_g), key="rnk_min_games")

    # ── Player leaderboard ─────────────────────────────────────────────────────
    lb_show = lb[lb["total_games"] >= min_g].copy()
    st.subheader(f"Player Leaderboard - {season_label} (Top {len(lb_show)})")
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

    # ── Season activity heatmap (All Seasons only) ─────────────────────────────
    if season_sel == _ALL_OPT and len(all_seasons) > 1:
        st.subheader("Season Activity - Top 20 Players")
        top_pids = set(lb.head(20)["player_id"])
        w_s = m_filtered[m_filtered["winnerId"].isin(top_pids)][["winnerId", "season"]].rename(columns={"winnerId": "player_id"})
        l_s = m_filtered[m_filtered["loserId"].isin(top_pids)][["loserId", "season"]].rename(columns={"loserId": "player_id"})
        both = pd.concat([w_s, l_s])
        both["player_name"] = both["player_id"].map(player_label_map)
        hm_pivot = (
            both.groupby(["player_name", "season"]).size()
            .unstack(fill_value=0)
            .reindex(columns=all_seasons, fill_value=0)
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
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Character meta ─────────────────────────────────────────────────────────
    st.subheader(f"Character Stats - {season_label}")
    cs = _char_meta(m_view, char_map, char_color_map)
    cs["top_players"] = [_top_players_str(m_view, cid, player_label_map) for cid in cs["char_id"]]

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
            title="Character Picked (all games)", xaxis_title="Games", yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
            height=max(320, len(cs) * 30 + 100),
            hoverlabel=dict(align="left"),
        )
        st.plotly_chart(fig_picks, use_container_width=True)

    # Win rate by character
    with col_b:
        cs_wr = cs.sort_values("win_rate", ascending=True)
        fig_wr_char = go.Figure(go.Bar(
            x=cs_wr["win_rate"], y=cs_wr["name"], orientation="h",
            marker=dict(
                color=cs_wr["win_rate"],
                colorscale="RdYlGn", cmin=0, cmax=100,
            ),
            customdata=cs_wr[["wins", "losses", "total"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Win Rate: %{x}%<br>"
                "Wins %{customdata[0]}  |  Losses %{customdata[1]}  |  Total %{customdata[2]}"
                "<extra></extra>"
            ),
            text=cs_wr["win_rate"].astype(str) + "%",
            textposition="outside",
        ))
        fig_wr_char.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
        fig_wr_char.update_xaxes(range=[0, 110])
        fig_wr_char.update_layout(
            title="Win Rate by Character", xaxis_title="Win Rate (%)", yaxis_title="",
            height=max(320, len(cs_wr) * 30 + 100),
            hoverlabel=dict(align="left"),
        )
        st.plotly_chart(fig_wr_char, use_container_width=True)

    # Character matchup win rate matrix
    agg = m_view.copy()
    agg["winner_char"] = agg["winnerCharacterId"].map(char_map)
    agg["loser_char"] = agg["loserCharacterId"].map(char_map)
    agg = agg.dropna(subset=["winner_char", "loser_char"])
    if not agg.empty:
        # Each matchup: winner_char beat loser_char
        mu = agg.groupby(["winner_char", "loser_char"]).size().reset_index(name="wins")
        mu_rev = agg.groupby(["loser_char", "winner_char"]).size().reset_index(name="losses")
        mu_rev.columns = ["winner_char", "loser_char", "losses"]
        mu_full = mu.merge(mu_rev, on=["winner_char", "loser_char"], how="outer").fillna(0)
        mu_full["total"] = mu_full["wins"] + mu_full["losses"]
        mu_full["win_rate"] = (mu_full["wins"] / mu_full["total"] * 100).round(1)
        pivot = mu_full.pivot(index="winner_char", columns="loser_char", values="win_rate")
        g_pivot = mu_full.pivot(index="winner_char", columns="loser_char", values="total").fillna(0).astype(int)

        char_order = sorted(cs["name"].dropna().tolist())
        pivot = pivot.reindex(index=char_order, columns=char_order)
        g_pivot = g_pivot.reindex(index=char_order, columns=char_order)

        text_vals = []
        for r in pivot.index:
            row_t = []
            for c in pivot.columns:
                wr = pivot.loc[r, c]
                g = g_pivot.loc[r, c]
                row_t.append(f"{wr:.0f}%<br>({g})" if pd.notna(wr) and g > 0 else "")
            text_vals.append(row_t)

        fig_mu = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            text=text_vals,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0, zmax=100,
            colorbar=dict(title="Win %"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Win Rate: %{z:.1f}%<extra></extra>",
        ))
        fig_mu.update_layout(
            title=f"Character Matchup Win Rates - {season_label}",
            xaxis_title="Opponent Character", yaxis_title="Player's Character",
            xaxis=dict(autorange="reversed"),
            yaxis=dict(autorange="reversed"),
            height=max(420, len(pivot) * 45 + 150),
            hoverlabel=dict(align="left"),
        )
        st.plotly_chart(fig_mu, use_container_width=True)
