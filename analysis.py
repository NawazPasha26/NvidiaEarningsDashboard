"""
@author - Nawaz Pasha
Updated on Thu Nov 20 17:16:47 2025 
Analysis update
"""

#analysis.py
#This module has the basic helper functions that are required for model development and analysis
#This also has the function to build the return model Total=Systematic + Idio

import numpy as np
import pandas as pd

def build_return_model(r, l, e, selected_factors, date_start, date_end):
    r_ = r[(r["Date"] >= date_start) & (r["Date"] <= date_end)].copy()
    l_ = l[(l["Date"] >= date_start) & (l["Date"] <= date_end)].copy()
    selected_factors = [c for c in selected_factors if c in r_.columns and c in l_.columns]
    if selected_factors:
        df = pd.merge(
            r_[["Date", "NVDA"] + selected_factors],
            l_[["Date"] + selected_factors],
            on="Date", suffixes=("_ret", "_beta"),
        )
        beta = df[[f"{f}_beta" for f in selected_factors]].to_numpy()
        ret = df[[f"{f}_ret" for f in selected_factors]].to_numpy()
        df["Factor_Pred_Return"] = (beta * ret).sum(axis=1)
        for f in selected_factors:
            df[f"{f}_contrib"] = df[f"{f}_beta"] * df[f"{f}_ret"]
    else:
        df = r_[["Date", "NVDA"]].copy()
        df["Factor_Pred_Return"] = 0.0
    df["Idio_Return"] = df["NVDA"] - df["Factor_Pred_Return"]
    df["NVDA_Cum"] = (1.0 + df["NVDA"]).cumprod() - 1.0
    return df, selected_factors, e.copy()


def get_next_trading_day(date_series: pd.Series, target_date: pd.Timestamp):
    idx = date_series.searchsorted(target_date, side="right")
    return date_series.iloc[idx] if idx < len(date_series) else None


def build_event_window(df, df_e, window=10):
    dates = df["Date"].reset_index(drop=True)
    valid, centers = [], []
    for d in df_e["EarningsDate"].dropna():
        d0 = get_next_trading_day(dates, d)
        if d0 is None:
            continue
        i0 = int(dates.searchsorted(d0))
        lo, hi = i0 - window, i0 + window
        if lo >= 0 and hi < len(dates):
            valid.append((lo, i0, hi))
            centers.append(i0)
    if not valid:
        return None
    idxs = np.arange(-window, window + 1)
    mats = {
        col: np.vstack([df[col].iloc[lo:hi + 1].to_numpy() for (lo, i0, hi) in valid])
        for col in ["NVDA", "Factor_Pred_Return", "Idio_Return"]
    }
    info = pd.DataFrame({
        "event_center_index": centers,
        "event_date": [df.loc[i0, "Date"] for (_, i0, _) in valid]
    })
    return mats, idxs, info


def safe_var(x):
    v = np.nanvar(x, ddof=1)
    return np.nan if v == 0 else v


def get_model_decomposition_stats(df):
    tot, fac, idi = df["NVDA"], df["Factor_Pred_Return"], df["Idio_Return"]
    sv_tot, sv_fac = safe_var(tot), safe_var(fac)
    r2 = np.nan if (np.isnan(sv_tot) or sv_tot == 0) else float(sv_fac / sv_tot)
    corr_tf = float(np.corrcoef(tot.fillna(0), fac.fillna(0))[0, 1]) if len(df) > 1 else np.nan
    hit_rate = float(np.mean(np.sign(tot) == np.sign(fac))) if len(df) > 0 else np.nan
    return tot, fac, idi, r2, corr_tf, hit_rate


def mask_triplet(idxs):
    return (idxs < 0), (idxs == 0), (idxs > 0)


def compute_event_averages(ev_out):
    if not ev_out:
        return None, None
    mats, idxs, info = ev_out
    return idxs, {
        "total": np.nanmean(mats["NVDA"], axis=0),
        "factor": np.nanmean(mats["Factor_Pred_Return"], axis=0),
        "idio": np.nanmean(mats["Idio_Return"], axis=0),
        "info": info
    }


def mean_over_mask(ts, m):
    return float(np.nanmean(ts[m]))


def compute_pre_post_means(avg, idxs):
    pre, ev, post = mask_triplet(idxs)
    a = avg
    return (
        mean_over_mask(a["total"], pre), mean_over_mask(a["total"], ev), mean_over_mask(a["total"], post),
        mean_over_mask(a["factor"], pre), mean_over_mask(a["factor"], ev), mean_over_mask(a["factor"], post),
        mean_over_mask(a["idio"], pre), mean_over_mask(a["idio"], ev), mean_over_mask(a["idio"], post)
    )


def compute_pre_post_abs_change(mats, idxs):
    pre, _, post = mask_triplet(idxs)
    pre_abs = np.nanmean(np.abs(mats["NVDA"][:, pre]), axis=1)
    post_abs = np.nanmean(np.abs(mats["NVDA"][:, post]), axis=1)
    return pre_abs, post_abs, (post_abs - pre_abs)