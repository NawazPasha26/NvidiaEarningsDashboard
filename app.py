"""
@author - Nawaz Pasha
Updated on Thu Nov 20 17:16:47 2025 
UI related style updates
"""

#ui.py
#This module is the main module where in the dashboard is put together
#All the module , helper functions built are imported here and used for model decomposition, data analysis
#In this module all the required stats, visuals, summaried are stitched together
#This is an extensive module where in teh data is explained in plain english and is rendered on the dashboard
#This dashboard has extensive analysis of NVIDIA returns decomposed by Factors and the issuer specific returns
#the dashboard has 7 tabs namely Overview, Data & EDA, Performance before/on/after Earnings, Total Return Decomposition, Volaitility Before vs After Earnings, Statistical Reliability & Limitations , Summary & Key Insights
#The dashboard has necessary filters put in place 
#This dashboard shows how the returns behave across Earning periods also how are the returns decomposed
#the necessary styles are put together and stitched for a user friendly dashboard experience
#The Earnings perios and Volatility filter is given for easy access.
#Extensive summary is built intoeach tab with proper visualizations, KPIs and stats,
#This has a default date filter that selects the data for 3 years Sep 2022 to Sep 2025, can be unchecked and any date within the data range can be selected.
#import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
#from typing import Optional

from constants import COLORS, PALETTE, TXT_EARNINGS_NOTE, TXT_VOL_EXPLAIN, TXT_DATA_EDA, TXT_LIMITS
from ui import use_global_css, control_panel_branding, df_show
from data import load_input_data
from analysis import (
    build_return_model, build_event_window, get_model_decomposition_stats,
    compute_event_averages, compute_pre_post_means, compute_pre_post_abs_change,
    mask_triplet)
from plots import (
    plot_timeseries_with_event_lines, plot_multitimeseries_with_event_lines,
    plot_avg_returns_selected, plot_cum_event_selected,
    apply_plotly_layout)

st.set_page_config(page_title="NVIDIA Earnings Analysis (2022–2025)", layout="wide")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("classic")



# ---- App start ----
use_global_css()
control_panel_branding()
r, l, e, all_factors, global_start, global_end = load_input_data()

# Sidebar controls


st.sidebar.header("Filters")

# Inform user about available date range
st.sidebar.markdown(
    f"Input Return Data is available only from {global_start.strftime('%b %Y')} to {global_end.strftime('%b %Y')} "
)

# Default date toggle
use_default_dates = st.sidebar.checkbox("Use dataset default Date Range (Oct 2022 - Sep 2025)", value=True)

# Prepare set of valid dates from the loaded return series 'r'
_valid_dates = set(r["Date"].dt.date)

if use_default_dates:
    sel_start = global_start
    sel_end = global_end
else:
    # Individual start and end date pickers
    try:
        start_date = st.sidebar.date_input(
            "Start date", global_start.date(),
            min_value=global_start.date(), max_value=global_end.date(), key="start_date"
        )
        end_date = st.sidebar.date_input(
            "End date", global_end.date(),
            min_value=global_start.date(), max_value=global_end.date(), key="end_date"
        )

        s_ts = pd.Timestamp(start_date)
        e_ts = pd.Timestamp(end_date)

        # Basic ordering validation
        if s_ts > e_ts:
            st.sidebar.error("Start date must be on or before End date. Please select proper valid dates.")
            sel_start, sel_end = global_start, global_end
        else:
            # Check that the exact selected dates exist in the input data (trade dates)
            if (s_ts.date() not in _valid_dates) or (e_ts.date() not in _valid_dates):
                # Show a friendly, prominent message in the main area (not only sidebar)
                st.error(
                    "Please select valid dates — the return input data contains trading dates only from "
                    "September-2022 to September-2025. Pick start/end dates that are present in the dataset."
                )
                st.sidebar.error("Selected start or end date is not present in the input data. Reverting to defaults.")
                sel_start, sel_end = global_start, global_end
            else:
                sel_start, sel_end = s_ts, e_ts
    except Exception:
        st.sidebar.error("Invalid date input. Using dataset default date range.")
        sel_start, sel_end = global_start, global_end

selected_factors = st.sidebar.multiselect(
    "Factors (affect factor-predicted return)", options=all_factors, default=all_factors
)
win = st.sidebar.slider("Event window (days before/after)", 5, 20, 10, 1)
roll = st.sidebar.slider("Rolling volatility window (days)", 10, 60, 20, 5)

## Return components filter
component_options = ["Total", "Systematic", "Idiosyncratic"]
show_components = st.sidebar.multiselect("Return components", component_options, default=component_options)

# Build model
df, selected_factors, df_e = build_return_model(r, l, e, selected_factors, sel_start, sel_end)
ev_out = build_event_window(df, df_e, window=win)

# Tabs
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Overview", "Data & EDA", "Earnings Event Analysis",
    "Factor-Based Return Decomposition", "Volatility Around Earnings Events",
    "Statistical Reliability & Limitations", "Summary & Key Insights"
])

# ---- Tab 1 ----
with t1:
    st.title("NVIDIA Earnings Analysis (2022–2025)")
    st.caption(TXT_EARNINGS_NOTE)

    in_range = df_e[(df_e["EarningsDate"] >= df["Date"].min()) & (df_e["EarningsDate"] <= df["Date"].max())]
    k1, k2, k3 = st.columns(3)
    k1.metric("Days in dataset", f"{df.shape[0]:,}")
    k2.metric("Factors used", f"{len(selected_factors)}")
    k3.metric("Earnings dates (covered)", f"{in_range.shape[0]}")

    st.subheader("Methodology")
    st.markdown(
        f"""**How we do it**:
1. Split each daily NVIDIA return into:  
   - **Factor-based** (sum of daily betas × factor returns)  
   - **Idiosyncratic** (NVIDIA minus factor-based)  
2. NVIDIA reports **after** the close → **day 0** is the **next trading day**.  
3. Analyze a configurable **±{win} days** window around earnings.  
4. Compare **Total, Factor, Idiosyncratic** returns and **volatility** changes."""
    )

    st.subheader("High-Level NVIDIA Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Daily Return (NVIDIA)**")
        fig = plot_timeseries_with_event_lines(df["Date"], df["NVDA"], "NVIDIA", COLORS["blue"],
                                  "NVIDIA Daily Returns", "Return", df_e["EarningsDate"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Cumulative Return (NVIDIA)**")
        fig = plot_timeseries_with_event_lines(df["Date"], df["NVDA_Cum"], "Cumulative", COLORS["green"],
                                  "NVIDIA Cumulative Return", "Cumulative Return", df_e["EarningsDate"],
                                  hover="%{x|%Y-%m-%d}<br>Cumulative: %{y:.2%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    # Dynamic decomposition series
    st.markdown("**Return Decomposition – Time Series (Daily)**")
    comp_to_series = {
        "Total": ("Total (NVIDIA)", df["NVDA"], 2.0, COLORS["blue"]),
        "Systematic": ("Systematic/Factor", df["Factor_Pred_Return"], 1.6, COLORS["orange"]),
        "Idiosyncratic": ("Idiosyncratic", df["Idio_Return"], 1.6, COLORS["green"]),
    }
    if show_components:
        series = [comp_to_series[c] for c in show_components if c in comp_to_series]
        fig = plot_multitimeseries_with_event_lines(
            df["Date"], series, df_e["EarningsDate"],
            "Total vs Systematic (Factors) vs Idiosyncratic – Daily Returns", "Daily Return"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one **Return component** in the sidebar to display this chart.")

# ---- Tab 2 ----
with t2:
    st.header("Data Quality & Exploratory Data Analysis")
    st.markdown(TXT_DATA_EDA)

    nulls, dups = df.isna().sum(), df.duplicated(subset=["Date"]).sum()
    z_nvda = np.abs(stats.zscore(df["NVDA"], nan_policy="omit"))
    z_idio = np.abs(stats.zscore(df["Idio_Return"], nan_policy="omit"))
    out_nvda, out_idio = int((z_nvda > 4).sum()), int((z_idio > 4).sum())

    c = st.columns(4)
    c[0].metric("Missing values (total)", int(nulls.sum()))
    c[1].metric("Duplicate dates", int(dups))
    c[2].metric("Outliers in NVIDIA (|z|>4)", out_nvda)
    c[3].metric("Outliers in Idiosyncratic (|z|>4)", out_idio)

    st.subheader("Summary Statistics (Daily Returns)")
    stat_cols = ["NVDA", "Factor_Pred_Return", "Idio_Return"]
    desc = df[stat_cols].describe().T
    if "count" in desc.columns:
        desc["count"] = desc["count"].astype(int)
    fmt_desc = {c: "{:.4f}" for c in desc.columns if c != "count"}
    fmt_desc["count"] = "{:d}"
    df_show(desc, fmt_desc)

    st.subheader("NVIDIA & Factor Returns – Time Series (Daily)")
    fac_ret_cols = [f"{f}_ret" for f in selected_factors if f"{f}_ret" in df.columns]
    if not fac_ret_cols:
        st.info("No factor return series available for the current selection.")
    else:
        ts = df[["Date", "NVDA"] + fac_ret_cols].copy()
        series = [("NVIDIA", ts["NVDA"], 2.0, COLORS["blue"])] + [
            (c.replace("_ret", ""), ts[c], 1.0, PALETTE[i % len(PALETTE)]) for i, c in enumerate(fac_ret_cols)
        ]
        fig = plot_multitimeseries_with_event_lines(ts["Date"], series, e["EarningsDate"],
                                        "NVIDIA vs Selected Factor Returns (Daily)", "Daily Return")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Thin dotted red guides indicate the first trading day after each earnings release (day 0).")

    st.subheader("Factor Return Correlations")
    fac_ret_cols_all = [f"{f}_ret" for f in selected_factors if f"{f}_ret" in df.columns]
    if len(fac_ret_cols_all) >= 2:
        corr = df[fac_ret_cols_all].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=[c.replace("_ret", "") for c in fac_ret_cols_all],
            y=[c.replace("_ret", "") for c in fac_ret_cols_all],
            colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="Corr"),
            hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>",
        ))
        fig = apply_plotly_layout(fig, title="Correlation Heatmap of Selected Factor Returns", bottom=60)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select 2 or more factors in the sidebar to view the correlation heatmap.")

# ==== Tab 3 ====
with t3:
    st.header("Performance Before / During / After Earnings")
    if not ev_out:
        st.warning("No earnings windows fully inside the sample for the selected window length / date range.")
    else:
        mats, idxs, event_info = ev_out
        idxs2, avg = compute_event_averages(ev_out)

        # Pre / Event / Post means
        (pre_tot, ev_tot, post_tot,
         pre_fac, ev_fac, post_fac,
         pre_idi, ev_idi, post_idi) = compute_pre_post_means(avg, idxs2)

        st.subheader("KPIs (Average Daily Returns)")
        c = st.columns(3)
        with c[0]:
            st.markdown(f"**Pre-window (avg over days -{win}..-1)**")
            st.metric("Total", f"{pre_tot:.3%}")
            st.metric("Factor", f"{pre_fac:.3%}")
            st.metric("Idio", f"{pre_idi:.3%}")
        with c[1]:
            st.markdown("**Event day (day 0)**")
            st.metric("Total", f"{ev_tot:.3%}")
            st.metric("Factor", f"{ev_fac:.3%}")
            st.metric("Idio", f"{ev_idi:.3%}")
        with c[2]:
            st.markdown(f"**Post-window (avg over days +1..+{win})**")
            st.metric("Total", f"{post_tot:.3%}")
            st.metric("Factor", f"{post_fac:.3%}")
            st.metric("Idio", f"{post_idi:.3%}")

        st.divider()
        st.markdown(
            f"""**How stock and factor returns behave around earnings (based on your current selections)**  
- **Before earnings (−{win}..−1)**: On average, **total** returns are {pre_tot:.3%}. Of this, ≈{pre_fac:.3%} is explained by **factors**, while the **idiosyncratic** component contributes ≈{pre_idi:.3%}.  
- **Event day (first trading day after results, day 0)**: Average **total** return is {ev_tot:.3%}. Only ≈{ev_fac:.3%} comes from **factors**, while **idiosyncratic** surprise contributes ≈{ev_idi:.3%}.  
- **After earnings (+1..+{win})**: Average **total** returns are {post_tot:.3%}, with ≈{post_fac:.3%} from **factors** and ≈{post_idi:.3%} from **idiosyncratic** effects."""
        )

        # Average returns line chart
        fig = plot_avg_returns_selected(idxs2, avg, ["Total", "Systematic", "Idiosyncratic"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Use the sidebar to change the window, date range, and selected factors (KPIs and the narrative update automatically).")

        # Boxplot section
        st.subheader("Boxplot: Returns on Earnings Day vs Non-Earnings Days")
        event_day_dates = pd.to_datetime(event_info["event_date"].dropna().unique())
        m = df["Date"].isin(event_day_dates)
        earn_r = df.loc[m, "NVDA"].dropna()
        non_r = df.loc[~m, "NVDA"].dropna()
        if earn_r.empty:
            st.info("No earnings-day returns available for the current selections.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=earn_r, name="Earnings Day (0)", boxmean=True,
                marker_color=COLORS["blue"], hovertemplate="%{y:.3%}<extra></extra>"
            ))
            fig.add_trace(go.Box(
                y=non_r, name="Non-Earnings Days", boxmean=True,
                marker_color=COLORS["gray"], hovertemplate="%{y:.3%}<extra></extra>"
            ))
            fig = apply_plotly_layout(fig, title="NVIDIA Returns: Earnings Day vs Non-Earnings Days",
                                      ytitle="Daily Return", bottom=80)
            st.plotly_chart(fig, use_container_width=True)

# ===== Tab 4 =====
with t4:
    st.header("Return Decomposition: Total = Systematic (Factors) + Idiosyncratic")
    st.markdown(
        "Each day we split NVIDIA’s total return into a **systematic factor piece** "
        "(what moves with market/sector/styles) and an **idiosyncratic piece** "
        "(company-specific surprise). The plots and tables below show how these parts evolve, "
        "including around earnings."
    )
    tot, fac, idi, r2, corr_tf, hit_rate = get_model_decomposition_stats(df)
    k = st.columns(5)
    k[0].metric("Variance explained (R²)", "n/a" if pd.isna(r2) else f"{r2:.1%}")
    k[1].metric("Corr(Total, Factor)", "n/a" if pd.isna(corr_tf) else f"{corr_tf:.2f}")
    k[2].metric("Hit rate (same sign)", "n/a" if pd.isna(hit_rate) else f"{hit_rate:.1%}")
    k[3].metric("Avg |Idiosyncratic|", f"{idi.abs().mean():.3%}")
    k[4].metric("Avg |Factor|", f"{fac.abs().mean():.3%}")

    with st.expander("What these KPIs mean?"):
        st.markdown(
            f"""
- **Variance explained (R²)** → “How much of NVIDIA’s daily ups/downs can be explained by the selected factors.”  
  - If R² is **high** (e.g., {('n/a' if pd.isna(r2) else f'{r2:.1%}')})→ NVIDIA is moving *with* the factors (market/sector/styles).  
  - If R² is **low** → company-specific news (earnings guidance, product news) matters more.  
**Impact on dashboard**: When R² is high, the factor lines you see in **Data & EDA** will track NVIDIA closely, and in the **Earnings tabs** you should expect a larger "factor" share of the move.
- **Correlation (Total vs Factor)** - *Do NVIDIA and the factor-predicted series move together day to day?*
  Value here is {('n/a' if pd.isna(corr_tf) else f'{corr_tf:.2f}')}.
  - **Closer to +1** → same direction moves are common.
  - **Near 0 or negative** → factors don't line up well with NVIDIA's day-to-day changes.
**Impact**: Higher correlation strengthens confidence that your chosen factor set is relevant for NVIDIA in this period.
- **Hit rate (same sign)** → *What fraction of days do total and Factor have the same sign?*
  Current hit rate is {('n/a' if pd.isna(hit_rate) else f'{hit_rate:.1%}')}.   
  - **Higher** → factors often get the **direction** right (even if the exact size differs).
**Impact**: A high hit rate makes the factor view more actionable for risk-hedging and expectation setting around events.
- **Avg Idiosyncratic vs Avg Factor** *(average *absolute* daily contribution)* → typical *size* of company-specific vs factor-driven moves.
 - Idiosyncratic ≈ **{idi.abs().mean():.3%}**; Factor ≈ **{fac.abs().mean():.3%}**.
**Impact**: If *Idiosyncratic > Factor*, then **earnings surprises** and NVIDIA-specific headlines dominate; if comparable, then **market/sector** risk is a big part of daily swings. This also guides how muchhedging( e.g., with sector/market ETFs) could reduce risk without removing your NVIDIA view. 
**How it ties together**
- Use these KPIs to interpret **"Earnings Event analysis"**: on **day 0**, if factor share is small and idiosyncratic is large, the move is mostly **company-specific**(true "surprise").
- In **"Top 5 Factors"** (below), check which factorsactually contributed the most- it should align with high R² / correlation periods.
- In **volatility Around Earnings"**, if post-earnings volatility falls, and R² stays high, the market likely "understood" the results quickly, leaving less uncertainity.
"""
        )

    # Essentials table
    rows = []
    if "Total" in show_components:
        rows.append(("Total (NVIDIA)", tot))
    if "Systematic" in show_components:
        rows.append(("Systematic (Factors)", fac))
    if "Idiosyncratic" in show_components:
        rows.append(("Idiosyncratic", idi))
    if rows:
        summ = pd.DataFrame({
            "Series": [r[0] for r in rows],
            "Mean": [r[1].mean() for r in rows],
            "Volatility (Std Dev)": [r[1].std() for r in rows],
            "Skew": [r[1].skew() for r in rows],
            "Kurtosis": [r[1].kurtosis() for r in rows],
            "Avg |Return|": [r[1].abs().mean() for r in rows],
        })
        df_show(summ, {"Mean": "{:.4%}", "Volatility (Std Dev)": "{:.4%}",
                       "Skew": "{:.2f}", "Kurtosis": "{:.2f}", "Avg |Return|": "{:.4%}"})
    else:
        st.info("Select at least one **Return component** in the sidebar to show the summary table.")

    st.subheader("Cumulative Average Around Earnings (within selected window)")
    if ev_out:
        idxs2, avg = compute_event_averages(ev_out)
        if show_components:
            fig = plot_cum_event_selected(idxs2, avg, show_components)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one **Return component** in the sidebar to show the cumulative chart.")
    else:
        st.info("No complete earnings windows inside the chosen date range to show the within-window cumulative chart.")

    st.subheader("Top 5 Factors by Contribution (for selected date range)")
    contrib_cols = [c for c in df.columns if c.endswith("_contrib")]
    if not contrib_cols:
        st.info("No factors selected – factor-predicted return is 0; all variation is idiosyncratic by construction."
                if not selected_factors else "No factor contributions available for the current selection.")
    else:
        contrib_sums = df[contrib_cols].sum(axis=0)
        contrib_abs = df[contrib_cols].abs().sum(axis=0)
        fac_total_sum = df["Factor_Pred_Return"].sum() if "Factor_Pred_Return" in df else 0.0
        top5_idx = contrib_abs.sort_values(ascending=False).head(5).index
        top5 = pd.DataFrame({
            "Factor": [c.replace("_contrib", "") for c in top5_idx],
            "Total contribution": contrib_sums.loc[top5_idx].values,
            "Total |Contribution|": contrib_abs.loc[top5_idx].values,
            "% of Factor- Pred Total (signed)": [
                np.nan if fac_total_sum == 0 else contrib_sums.loc[i] / fac_total_sum for i in top5_idx
            ],
            "Avg Daily contribution": df[top5_idx].mean().values,
            "Contribution Volatility (Std)": df[top5_idx].std().values,
        })
        df_show(top5, {
            "Total contribution": "{:.4%}",
            "Total |Contribution|": "{:.4%}",
            "% of Factor- Pred Total (signed)": "{:.1%}",
            "Avg Daily contribution": "{:.4%}",
            "Contribution Volatility (Std)": "{:.4%}",
        })
        fig = go.Figure(go.Bar(
            x=top5["Factor"], y=top5["Total contribution"],
            marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in top5["Total contribution"]],
            customdata=np.stack([
                top5["Total contribution"], top5["Avg Daily contribution"],
                top5["Contribution Volatility (Std)"], top5["% of Factor- Pred Total (signed)"]
            ], axis=-1),
            hovertemplate=(
                "Factor: %{x}<br>Total Contribution: %{y:.4%}"
                "<br>Avg Daily: %{customdata[1]:.4%}"
                "<br>Vol (Std): %{customdata[2]:.4%}"
                "<br>% of Factor-Pred Total (signed): %{customdata[3]:.1%}<extra></extra>"
            )
        ))
        fig = apply_plotly_layout(fig, title="Top 5 Factors – Total Signed Contribution (for selected date range)",
                                  ytitle="Total Contribution", bottom=80)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factors in use")
    st.write(", ".join(selected_factors) if selected_factors else
             "No factors selected – factor-predicted return is 0; all variation is idiosyncratic by construction.")

# ==== Tab 5 ====
with t5:
    st.header("Volatility Before vs After Earnings")
    st.markdown(TXT_VOL_EXPLAIN)
    ann = np.sqrt(252)
    df[f"NVDA_RollingVol_{roll}"] = df["NVDA"].rolling(roll).std() * ann
    df[f"Idio_RollingVol_{roll}"] = df["Idio_Return"].rolling(roll).std() * ann

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rolling Volatility: NVIDIA (annualized)**")
        fig = plot_timeseries_with_event_lines(
            df["Date"], df[f"NVDA_RollingVol_{roll}"], f"{roll}-day Vol", COLORS["purple"],
            f"NVIDIA Rolling Volatility ({roll}-day, annualized)", "Annualized Volatility",
            df_e["EarningsDate"], "%{x|%Y-%m-%d}<br>Ann. Vol: %{y:.2%}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Rolling Volatility: Idiosyncratic (annualized)**")
        fig = plot_timeseries_with_event_lines(
            df["Date"], df[f"Idio_RollingVol_{roll}"], f"{roll}-day Idio Vol", COLORS["brown"],
            f"Idiosyncratic Rolling Volatility ({roll}-day, annualized)", "Annualized Volatility",
            df_e["EarningsDate"], "%{x|%Y-%m-%d}<br>Ann. Vol: %{y:.2%}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pre vs Post Volatility (per event)")
    if ev_out:
        mats, idxs, event_info = ev_out
        pre_abs, post_abs, change = compute_pre_post_abs_change(mats, idxs)
        vol_df = pd.DataFrame({
            "EventDate": event_info["event_date"],
            "PrevAvgAbsRet": pre_abs, "PostAvgAbsRet": post_abs, "Change": change
        }).sort_values("EventDate")
        df_show(vol_df, {"PrevAvgAbsRet": "{:.4f}", "PostAvgAbsRet": "{:.4f}", "Change": "{:.4f}"})
        fig = go.Figure(go.Bar(
            x=vol_df["EventDate"].dt.strftime("%Y-%m-%d"), y=vol_df["Change"],
            marker_color=np.where(vol_df["Change"] > 0, COLORS["green"], COLORS["red"]),
            customdata=np.stack([vol_df["PrevAvgAbsRet"], vol_df["PostAvgAbsRet"]], axis=1),
            hovertemplate=(
                "Event: %{x}<br>Post – Pre: %{y:.4f}"
                "<br>PrevAvgAbsRet: %{customdata[0]:.4f}"
                "<br>PostAvgAbsRet: %{customdata[1]:.4f}<extra></extra>"
            )
        ))
        fig = apply_plotly_layout(fig, title="Change in Avg |Daily Return| (Post – Pre)",
                                  ytitle="Avg Return Change (post – pre)", bottom=90)
        st.plotly_chart(fig, use_container_width=True)

        pre_mask, day0_mask, post_mask = mask_triplet(idxs)
        pre_abs_mean = float(np.nanmean(np.nanmean(np.abs(mats["NVDA"][:, pre_mask]), axis=1)))
        post_abs_mean = float(np.nanmean(np.nanmean(np.abs(mats["NVDA"][:, post_mask]), axis=1)))
        day0_abs_vals = np.abs(mats["NVDA"][:, day0_mask]).flatten()
        day0_abs_mean = float(np.nanmean(day0_abs_vals)) if day0_abs_vals.size else np.nan

        pre_idio_abs_mean = float(np.nanmean(np.nanmean(np.abs(mats["Idio_Return"][:, pre_mask]), axis=1)))
        post_idio_abs_mean = float(np.nanmean(np.nanmean(np.abs(mats["Idio_Return"][:, post_mask]), axis=1)))
        day0_idio_abs_vals = np.abs(mats["Idio_Return"][:, day0_mask]).flatten()
        day0_idio_abs_mean = float(np.nanmean(day0_idio_abs_vals)) if day0_idio_abs_vals.size else np.nan

        post_vs_pre = "smaller" if post_abs_mean < pre_abs_mean else ("larger" if post_abs_mean > pre_abs_mean else "about the same")

        st.subheader("How earnings moves tend to behave?")
        st.markdown(
            f"""
- **Before earnings (days −{win}..−1)**: The typical **daily move size** (absolute return) is about **{pre_abs_mean:.2%}** for NVIDIA,with the company-specific piece around **{pre_idio_abs_mean:.2%}**.      
- *Interpretation*: This is the *waiting period*; moves reflect regular market activity plus any pre-announcement chatter.    
- **On the first trading day after the announcement (day 0)**: The **one-day move** averages **{day0_abs_mean:.2%}**, and the **idiosyncratic** share  
  averages **{day0_idio_abs_mean:.2%}**.
  *Interpretation*: This is usually the *biggest* jump because the market digests the new information in one go.    
- **After earnings (days +1..+{win})**: the typical **daily move size** settles to **{post_abs_mean:.2%}**,  which is **{post_vs_pre}** than the prewindow on average.
"""
        )
    else:
        st.info("No complete earnings windows inside the selected date range and window.")

# ==== Tab 6 ====
with t6:
    st.header("Statistical Reliability & Limitations")
    st.subheader("What we check in this dashboard")
    st.markdown(
        "- **Event-day average return ≠ 0?** We test the **Idiosyncratic** part and the **Total** NVIDIA return on the "
        "**first trading day after earnings (day 0)**. *Why:* It tells us if day-0 moves are typically non-zero across events.\n"
        "- **Does volatility change after earnings?** We compare the **average absolute daily return** per event **before** vs "
        "**after** earnings (t-test). *Why:* It shows whether typical daily move size rises or falls post results. \n"
        "- We display **p-values** for these tests. **A smaller p (<0.05)** suggests the effect is unlikely due to chance."
    )
    if ev_out:
        mats, idxs, _ = ev_out
        pre, ev_mask, post = mask_triplet(idxs)
        event_idio = mats["Idio_Return"][:, ev_mask].flatten()
        event_total = mats["NVDA"][:, ev_mask].flatten()
        pre_abs, post_abs, _ = compute_pre_post_abs_change(mats, idxs)

        t1_stat, t1_p = stats.ttest_1samp(event_idio[~np.isnan(event_idio)], popmean=0.0)
        t3_stat, t3_p = stats.ttest_1samp(event_total[~np.isnan(event_total)], popmean=0.0)
        t2_stat, t2_p = stats.ttest_ind(post_abs, pre_abs, equal_var=False, nan_policy="omit")

        c = st.columns(3)
        c[0].metric("Event-day **Idiosyncratic** mean ≠ 0", f"{float(t1_p):.4f}")
        c[1].metric("Post vs Pre **Avg |return|** (t-test)", f"{float(t2_p):.4f}")
        c[2].metric("Event-day **Total** mean ≠ 0", f"{float(t3_p):.4f}")
    else:
        st.info("No complete earnings windows inside your selected date range and window. "
                "Widen the date range or reduce the window to include more events.")

    st.subheader("Limitations")
    st.markdown(TXT_LIMITS)

# ==== Tab 7 ====
with t7:
    st.header("Summary & Key Insights")
    st.markdown(
        "This page tells you **what actually happened** around NVIDIA’s earnings. "
        "We summarise performance **before**, **on**, and **after** earnings, to show how much was driven by "
        "**factors vs. company news**, explain **volatility**, list **Strategy ideas**, and note "
        "**statistical reliability & limits**."
    )

    tot, fac, idi, r2, corr_tf, hit_rate = get_model_decomposition_stats(df)
    avg_abs_idio, avg_abs_fac = float(idi.abs().mean()), float(fac.abs().mean())

    pre_tot = ev_tot = post_tot = pre_fac = ev_fac = pre_idi = post_fac = ev_idi = post_idi = np.nan
    n_events = 0
    pct_up = pct_down = np.nan

    def share_abs(a, b):
        denom = abs(a) + abs(b)
        return np.nan if (denom == 0 or pd.isna(a) or pd.isna(b)) else abs(a) / denom

    if ev_out:
        mats, idxs, event_info = ev_out
        n_events = mats["NVDA"].shape[0]
        idxs2, avg = compute_event_averages(ev_out)
        (pre_tot, ev_tot, post_tot,
         pre_fac, ev_fac, post_fac,
         pre_idi, ev_idi, post_idi) = compute_pre_post_means(avg, idxs2)
        pre_abs, post_abs, change = compute_pre_post_abs_change(mats, idxs)
        if n_events > 0:
            pct_up, pct_down = float(np.mean(change > 0)), float(np.mean(change < 0))

    st.subheader("1) Performance of NVIDIA During Earnings")
    if not ev_out or n_events == 0:
        st.markdown("We don’t have complete earnings windows inside the selected date range."
                    "Use the slider to widen the date range or reduce the window to include more events.")
    else:
        perf = pd.DataFrame({
            "Window": [f"Pre (−{win}..−1)", "Event day (0)", f"Post (+1..+{win})"],
            "Avg Total Return": [pre_tot, ev_tot, post_tot],
            "Avg Factor Return": [pre_fac, ev_fac, post_fac],
            "Avg Idiosyncratic Return": [pre_idi, ev_idi, post_idi],
            "Factor share of move": [share_abs(pre_fac, pre_idi), share_abs(ev_fac, ev_idi), share_abs(post_fac, post_idi)],
        })
        df_show(perf, {"Avg Total Return": "{:.3%}", "Avg Factor Return": "{:.3%}",
                       "Avg Idiosyncratic Return": "{:.3%}", "Factor share of move": "{:.1%}"})

        st.markdown(
            f"- **Before earnings**: Total = {pre_tot:.3%}, Factor = {pre_fac:.3%}, Idio = {pre_idi:.3%} "
            f"(≈{(share_abs(pre_fac, pre_idi) if not np.isnan(share_abs(pre_fac,pre_idi)) else 0):.0%} of the typical move from factors).\n"
            f"- **On earnings day**: Total = {ev_tot:.3%}, Factor = {ev_fac:.3%}, Idio = {ev_idi:.3%} "
            f"(≈{(share_abs(ev_fac, ev_idi) if not np.isnan(share_abs(ev_fac,ev_idi)) else 0):.0%} factor share, rest is company-specific surprise).\n"
            f"- **After earnings**: Total = {post_tot:.3%}, Factor = {post_fac:.3%}, Idio = {post_idi:.3%} "
            f"(≈{(share_abs(post_fac, post_idi) if not np.isnan(share_abs(post_fac, post_idi)) else 0):.0%} factor share)."
        )

        median_abs_total = float(df["NVDA"].abs().median()) if not df.empty else np.nan
        median_abs_fac = float(df["Factor_Pred_Return"].abs().median()) if "Factor_Pred_Return" in df else np.nan
        median_abs_idio = float(df["Idio_Return"].abs().median()) if "Idio_Return" in df else np.nan

        day0_fac_share = day0_idio_share = day0_total_abs = np.nan
        pct_days_factor_larger = np.nan
        if ev_out:
            day0_mask = (idxs2 == 0)
            if day0_mask.any():
                day0_fac = mats["Factor_Pred_Return"][:, day0_mask].flatten()
                day0_idio = mats["Idio_Return"][:, day0_mask].flatten()
                day0_tot = mats["NVDA"][:, day0_mask].flatten()
                day0_total_abs = float(np.nanmean(np.abs(day0_tot)))
                denom = np.abs(day0_fac) + np.abs(day0_idio)
                valid = denom > 0
                if valid.any():
                    day0_fac_share = float(np.nanmean(np.abs(day0_fac[valid]) / denom[valid]))
                    day0_idio_share = 1.0 - day0_fac_share
                    pct_days_factor_larger = float(np.mean(df["Factor_Pred_Return"].abs() > df["Idio_Return"].abs()))

        st.subheader("2) Factor-Based Return Decomposition")
        st.markdown(
            f"""
- **How much do factors typically matter?** On an average day, the **absolute** factor move is about **{avg_abs_fac:.2%}**  vs **{avg_abs_idio:.2%}** idiosyncratic (mean of absolute values).  
  For robustness, the **median** sizes are ~**{median_abs_fac:.2%}** (factor) vs ~**{median_abs_idio:.2%}** (idiosyncratic), compared to median total move **{median_abs_total:.2%}**.
- **When do factors dominate?** Across most days in your selection, factors were **larger than idiosyncratic** on roughly **{(pct_days_factor_larger if not np.isnan(pct_days_factor_larger) else 0):.0%}** of days (by absolute size).
- **On Earnings day (day 0), what drives the move?** the average **one-day total move** is about
  **{(day0_total_abs if not np.isnan(day0_total_abs) else 0):.2%}**. Of that, **factors explain** ~**{(day0_fac_share if not np.isnan(day0_fac_share) else 0):.0%}**,  
  and **company-specific (idiosyncratic)**  explains ~**{(day0_idio_share if not np.isnan(day0_idio_share) else 0):.0%}** (using absolute contributions to avoid sign cancellation).
- **Reading R², Correlation, Hit-rate together**:
  - Higher **R²** + high **correlation** + high **hit-rate** → the factor model is capturing NVIDIA’s environment well; hedging market/sector risk should meaningfully reduce variance without removing the view on the stock.
  - Low **R²** with modest **correlation/hit-rate** → day-to-day and especially earnings day moves are more **stock specific**;  focus more on fundamentals.      
"""
        )

    st.markdown(
        f"- **R² (variance explained by factors)**: {('n/a' if pd.isna(r2) else f'{r2:.1%}')}\n"
        f"- **Correlation (Total vs Factor)**: {('n/a' if pd.isna(corr_tf) else f'{corr_tf:.2f}')}\n"
        f"- **Hit rate (same sign)**: {('n/a' if pd.isna(hit_rate) else f'{hit_rate:.1%}')}\n"
        f"- **Avg Idiosyncratic**: {avg_abs_idio:.3%} , **Avg Factor**: {avg_abs_fac:.3%}"
    )

    if ev_out and n_events > 0:
        pre_mask, day0_mask, post_mask = mask_triplet(idxs2)
        mats_total, mats_idio = mats["NVDA"], mats["Idio_Return"]
        pre_event_abs = np.nanmean(np.abs(mats_total[:, pre_mask]), axis=1)
        post_event_abs = np.nanmean(np.abs(mats_total[:, post_mask]), axis=1)
        day0_abs = np.abs(mats_total[:, day0_mask].flatten())

        calm_pct = float(np.mean(post_event_abs < pre_event_abs))
        heat_pct = float(np.mean(post_event_abs > pre_event_abs))
        pre_med = float(np.nanmedian(pre_event_abs))
        post_med = float(np.nanmedian(post_event_abs))
        day0_med = float(np.nanmedian(day0_abs))

        day0_fac = np.abs(mats["Factor_Pred_Return"][:, day0_mask]).flatten()
        day0_idi = np.abs(mats_idio[:, day0_mask]).flatten()
        denom = day0_fac + day0_idi
        valid = denom > 0
        idio_share_med = float(np.nanmedian(day0_idi[valid] / denom[valid])) if valid.any() else np.nan
        idio_share_mean = float(np.nanmean(day0_idi[valid] / denom[valid])) if valid.any() else np.nan
        max_day0 = float(np.nanmax(day0_abs)) if day0_abs.size else np.nan

        # readable comparison descriptor
        day0_vs_pre = "much larger" if day0_med > pre_med * 1.5 else ("larger" if day0_med > pre_med * 1.1 else "only slightly larger")

        st.subheader("3) Volatility around earnings")
        st.markdown(
            f"""
- **How big is day 0 vs a typical day?** Median **day 0 move** ≈ **{day0_med:.2%}**, compared with median **pre-window daily move** ≈ **{pre_med:.2%}** and **postwindow** ≈ **{post_med:.2%}**. → Day 0 is usually **{day0_vs_pre}** than a regular pre-earnings day.  
- **What fraction of events calm down after results?** **{calm_pct:.0%}** of events show **lower typical daily moves** after earnings (post < pre), while **{heat_pct:.0%}** **heat up** (post > pre).
- **What truly drives day-0?** The **idiosyncratic share** of the day-0 move (absolute) is about **{idio_share_mean:.0%}** on average and (median **{idio_share_med:.0%}**). → Most of the day 0 reaction is typically **company-specific**, not broad market factors.  
- **Biggest single day-0 swing** in your selected window is about **{max_day0:.2%}** (absolute) – useful for stress-testing.
"""
        )
    else:
        st.markdown("Event-aligned volatility stats need at least one complete earnings window. "
                    "Adjust the date range or window length to populate these numbers.")

    st.subheader("4) Possible Strategies")
    st.markdown(
        "- **Limit risk around earnings**: Consider reducing your position or going flat before results if you want to avoid surprises.\n"
        "- **Hedge market exposure**: If *factors* swing, use ETFs or futures to offset market/sector risk while keeping NVIDIA shares.\n"
        "- **Post-earnings trend**: Decide whether to follow or fade the day-0 move based on fundamentals, and always manage risk.\n"
        "- **Factor selection**: Avoid piling into highly correlated factors; check the **Top 5 Factors** section in Factor-Based Return decomposition tab for guidance."
    )

    st.subheader("5) Statistical reliability & limitations")
    if ev_out:
        mats, idxs, _ = ev_out
        pre_abs, post_abs, _ = compute_pre_post_abs_change(mats, idxs)
        ev_idio = mats["Idio_Return"][:, idxs == 0].flatten()
        ev_total = mats["NVDA"][:, idxs == 0].flatten()
        t1_stat, t1_p = stats.ttest_1samp(ev_idio[~np.isnan(ev_idio)], popmean=0.0)
        t2_stat, t2_p = stats.ttest_ind(post_abs, pre_abs, equal_var=False, nan_policy="omit")
        t3_stat, t3_p = stats.ttest_1samp(ev_total[~np.isnan(ev_total)], popmean=0.0)
        st.markdown(
            f"- **Event day idiosyncratic mean ≠ 0 (t-test)** p-value: **{float(t1_p):.4f}**\n"
            f"- **Post vs pre avg |return| (t-test)** p-value: **{float(t2_p):.4f}**\n"
            f"- **Event-day total mean ≠ 0 (t-test)** p-value: **{float(t3_p):.4f}**\n\n"
            f"**Limitations**: \n{TXT_LIMITS}"
        )
    else:
        st.markdown("We can't run reliability tests without complete windows. ")
