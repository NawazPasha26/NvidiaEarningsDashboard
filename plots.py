"""
@author - Nawaz Pasha
Updated on Thu Nov 20 17:16:47 2025 
Plotly Graph & visual helper, plotter Functions
"""

#plots.py
#This module consists of all the helper function that will be required to plts visuals using plotly

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from constants import COLORS
from analysis import get_next_trading_day

# ---- Plot helpers ----
def apply_plotly_layout(fig, title="", ytitle=None, xtitle=None, bottom=80, top=56, legend_bottom=True):
    # Position legend outside (below) when legend_bottom=True to avoid overlap
    if legend_bottom:
        legend_dict = dict(orientation="h",x=0.0,xanchor="left",y=-0.18,yanchor="top",traceorder="normal",bgcolor="rgba(0,0,0,0)",bordercolor="rgba(0,0,0,0)")
        # Ensure a larger bottom margin so legend area has room
        margin_bottom = max(bottom, 140)
    else:
        legend_dict = dict(orientation="h",x=0.0,xanchor="left",y=1.02,yanchor="bottom",traceorder="normal",bgcolor="rgba(0,0,0,0)",bordercolor="rgba(0,0,0,0)")
        margin_bottom = bottom
    fig.update_layout(
        title=dict(text=title, y=0.96, x=0.01, xanchor="left", yanchor="top"),
        autosize=True,
        margin=dict(l=10, r=10, t=top, b=margin_bottom),
        legend=legend_dict
    )
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    return fig


def _ensure_series(x):
    return x.reset_index(drop=True) if isinstance(x, pd.Series) else pd.Series(x).reset_index(drop=True)


def get_event_reaction_days(date_series, earnings_dates):
    d = _ensure_series(date_series)
    out = []
    for ed in earnings_dates.dropna():
        d0 = get_next_trading_day(d, pd.Timestamp(ed))
        if d0 is not None:
            out.append(pd.Timestamp(d0))
    return sorted(set(out))


def add_earnings_day_lines(fig, x_dates, earnings_dates, color="red"):
    ev_days = get_event_reaction_days(x_dates, earnings_dates)
    if not ev_days:
        return
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    for d in ev_days:
        shapes.append(dict(type="line", xref="x", yref="paper",
                           x0=d, x1=d, y0=0, y1=1,
                           line=dict(color=color, width=1, dash="dot")))
    fig.update_layout(shapes=shapes)


def add_earnings_day_markers(fig, x_dates, y_values, earnings_dates, color="red", name="Earnings (day 0)"):
    xd, ys = _ensure_series(x_dates), _ensure_series(y_values)
    xs_idx = {pd.Timestamp(x): i for i, x in enumerate(xd)}
    pts_x, pts_y, pts_text = [], [], []
    for d in get_event_reaction_days(xd, earnings_dates):
        i = xs_idx.get(pd.Timestamp(d))
        if i is not None and i < len(ys) and pd.notna(ys.iloc[i]):
            pts_x.append(d)
            pts_y.append(ys.iloc[i])
            pts_text.append(f"Earnings reaction day<br>{d.strftime('%Y-%m-%d')}<br>Value: {ys.iloc[i]:.4%}")
    if pts_x:
        fig.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="markers", name=name,
            marker=dict(color=color, size=6, symbol="x"),
            hovertemplate="%{text}<extra></extra>", text=pts_text, showlegend=False
        ))


def plot_timeseries_with_event_lines(x, y, name, color, title, ytitle, earnings,
                        hover="%{x|%Y-%m-%d}<br>%{y:.3%}<extra></extra>"):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name=name,
                               line=dict(color=color), hovertemplate=hover))
    add_earnings_day_lines(fig, x, earnings, COLORS["red"])
    add_earnings_day_markers(fig, x, y, earnings, COLORS["red"])
    return apply_plotly_layout(fig, title=title, ytitle=ytitle)


def plot_multitimeseries_with_event_lines(x, series, earnings, title, ytitle):
    fig = go.Figure()
    for name, y, width, color in series:
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(width=width, color=color),
            meta=name, hovertemplate="%{x|%Y-%m-%d}<br><b>%{meta}</b>: %{y:.3%}<extra></extra>",
        ))
    add_earnings_day_lines(fig, x, earnings, COLORS["red"])
    if series:
        add_earnings_day_markers(fig, x, series[0][1], earnings, COLORS["red"])
    return apply_plotly_layout(fig, title=title, ytitle=ytitle)


def plot_avg_returns_selected(idxs, avg_dict, selected):
    fig = go.Figure()
    mapping = {
        "Total": ("total", "Total (NVIDIA)", COLORS["blue"]),
        "Systematic": ("factor", "Factor-based", COLORS["orange"]),
        "Idiosyncratic": ("idio", "Idiosyncratic", COLORS["green"]),
    }
    for label in selected:
        if label not in mapping:
            continue
        key, disp, col = mapping[label]
        fig.add_trace(go.Scatter(
            x=idxs, y=avg_dict[key], name=disp, line=dict(color=col),
            hovertemplate="Day %{x}: %{y:.3%}<extra></extra>"
        ))
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")
    return apply_plotly_layout(
        fig,
        title="Average Daily Returns Around Earnings (Aligned)",
        xtitle="Event day (0 = first trading day after earnings)",
        ytitle="Average Return",
        bottom=90
    )


def plot_cum_event_selected(idxs, avg_dict, selected):
    fig = go.Figure()
    mapping = {
        "Total": ("total", "Cumulative Total", COLORS["blue"]),
        "Systematic": ("factor", "Cumulative Systematic (Factors)", COLORS["orange"]),
        "Idiosyncratic": ("idio", "Cumulative Idiosyncratic", COLORS["green"]),
    }
    for label in selected:
        if label not in mapping:
            continue
        key, disp, col = mapping[label]
        cum = np.cumsum(avg_dict[key])
        fig.add_trace(go.Scatter(
            x=idxs, y=cum, name=disp, line=dict(color=col),
            hovertemplate="Day %{x}: %{y:.3%}<extra></extra>"
        ))
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")
    return apply_plotly_layout(
        fig,
        title="Cumulative Average Returns Around Earnings (Decomposed)",
        xtitle="Event day",
        ytitle="Cumulative Avg Return (within window)",
        bottom=90
    )
