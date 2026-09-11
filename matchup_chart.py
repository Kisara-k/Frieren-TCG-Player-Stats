"""Shared character-matchup chart used by both application views."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_character_matchup_chart(
    matches: pd.DataFrame,
    title: str,
    *,
    player_col: str = "player_char_name",
    opponent_col: str = "opp_char_name",
    result_col: str = "result",
) -> go.Figure:
    """Build a consistently ordered matchup heatmap with marginal W/L bars."""
    data = matches[[player_col, opponent_col, result_col]].dropna().copy()
    character_order = sorted(
        set(data[player_col]).union(data[opponent_col]),
        key=lambda name: str(name).casefold(),
    )

    aggregate = data.groupby([player_col, opponent_col]).agg(
        Games=(result_col, "count"),
        Wins=(result_col, lambda values: (values == "Win").sum()),
    ).reset_index()
    aggregate["Losses"] = aggregate["Games"] - aggregate["Wins"]
    aggregate["WinRate"] = aggregate["Wins"] / aggregate["Games"] * 100

    def matrix(column, fill_value=None):
        result = aggregate.pivot(index=player_col, columns=opponent_col, values=column)
        result = result.reindex(index=character_order, columns=character_order)
        return result.fillna(fill_value) if fill_value is not None else result

    win_rate = matrix("WinRate")
    games = matrix("Games", 0).astype(int)
    wins = matrix("Wins", 0).astype(int)
    losses = matrix("Losses", 0).astype(int)

    labels = [
        [f"{rate:.0f}%<br>({games.loc[row, col]})" if pd.notna(rate) else ""
         for col, rate in win_rate.loc[row].items()]
        for row in win_rate.index
    ]
    row_wins, row_losses = wins.sum(axis=1), losses.sum(axis=1)
    col_wins, col_losses = wins.sum(axis=0), losses.sum(axis=0)
    positions = list(range(len(character_order)))

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.15, 0.85], row_heights=[0.85, 0.15],
        horizontal_spacing=0.015, vertical_spacing=0.015,
    )
    fig.add_trace(go.Heatmap(
        z=win_rate.values, x=positions, y=positions,
        text=labels, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=0, zmax=100, zmid=50,
        colorbar=dict(title="Win %", x=1.12),
        hovertemplate=(
            "<b>%{y}</b> vs <b>%{x}</b><br>"
            "Win Rate: %{z:.1f}%<br>Games: %{customdata}<extra></extra>"
        ),
        customdata=games.values,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=row_wins.values, y=positions, orientation="h",
        marker_color="#2ecc71", name="Wins", legendgroup="Wins",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=row_losses.values, y=positions, orientation="h",
        marker_color="#e74c3c", name="Losses", legendgroup="Losses",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=positions, y=col_wins.values,
        marker_color="#2ecc71", name="Wins", legendgroup="Wins", showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=positions, y=col_losses.values,
        marker_color="#e74c3c", name="Losses", legendgroup="Losses", showlegend=False,
    ), row=2, col=2)

    # The marginal plots label the left/bottom; the heatmap repeats the same
    # alphabetical axes on the top/right. scaleanchor keeps every cell square
    # as Plotly responsively fits the chart to its available container width.
    fig.update_yaxes(
        tickmode="array", tickvals=positions, ticktext=character_order,
        range=[len(character_order) - 0.5, -0.5],
        showticklabels=True, title_text="Player's Character", row=1, col=1,
    )
    fig.update_xaxes(
        autorange="reversed", showticklabels=True, showgrid=True, dtick=50,
        gridcolor="rgba(128,128,128,0.25)", zeroline=False,
        row=1, col=1,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=positions, ticktext=character_order,
        range=[-0.5, len(character_order) - 0.5], constrain="domain",
        side="top", showticklabels=True, tickangle=-45, row=1, col=2,
    )
    fig.update_yaxes(
        tickmode="array", tickvals=positions, ticktext=character_order,
        range=[len(character_order) - 0.5, -0.5], constrain="domain",
        side="right", showticklabels=True, scaleanchor="x2", scaleratio=1,
        row=1, col=2,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=positions, ticktext=character_order,
        range=[-0.5, len(character_order) - 0.5],
        showticklabels=True, tickangle=-45, title_text="Opponent's Character",
        row=2, col=2,
    )
    fig.update_yaxes(
        showticklabels=True, showgrid=True, dtick=50,
        gridcolor="rgba(128,128,128,0.25)", zeroline=False,
        row=2, col=2,
    )
    fig.update_layout(
        title=title, barmode="stack", autosize=True,
        height=max(620, len(character_order) * 47 + 260),
        margin=dict(t=140, r=180, b=150, l=160),
        hoverlabel=dict(align="left"), legend=dict(orientation="h"),
    )
    return fig
