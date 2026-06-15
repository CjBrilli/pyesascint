# ============================================================
# pride_transfer_analysis.py
# Uniform DSN–PRIDE transfer analysis
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit

from src.io_utils import load_horizons_daily_sep


# ============================================================
# Horizons / SEP handling
# ============================================================

def find_horizons_file(project_root, data_root, year):
    candidates = [
        data_root / "Scint_analysis" / f"vex_{year}.txt",
        data_root / "inputs" / f"vex_{year}.txt",
        project_root / "inputs" / f"vex_{year}.txt",
        project_root.parent / "Scint_analysis" / f"vex_{year}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def add_daily_elongation_to_bins(binned, project_root, data_root, year):
    horizons_file = find_horizons_file(project_root, data_root, year)
    binned = binned.copy()

    if horizons_file is None:
        print(f"No Horizons file found for {year}")
        binned["elongation_deg"] = np.nan
        return binned

    horizons_daily = load_horizons_daily_sep(horizons_file)
    horizons_daily = horizons_daily[["day", "elongation_deg"]].copy()
    horizons_daily["day"] = pd.to_datetime(horizons_daily["day"]).dt.floor("D")

    binned["timestamp"] = pd.to_datetime(binned.index)
    binned["day"] = binned["timestamp"].dt.floor("D")

    binned = binned.merge(
        horizons_daily,
        on="day",
        how="left",
    )

    binned = binned.set_index("timestamp").sort_index()

    print(f"{year} Horizons:", horizons_file)
    print(f"{year} missing elongation:", binned["elongation_deg"].isna().sum())

    return binned


# ============================================================
# Core statistics
# ============================================================

def safe_spearman(df):
    sub = df[["pride_scint_rad", "dsn_sigma_phi_rad"]].dropna()

    if len(sub) < 3:
        return np.nan, np.nan

    r, p = stats.spearmanr(
        sub["pride_scint_rad"],
        sub["dsn_sigma_phi_rad"],
    )

    return r, p


def simple_signal_correlations(df):
    sub = df[["pride_scint_rad", "dsn_sigma_phi_rad"]].dropna()

    if len(sub) < 3:
        return {
            "n": len(sub),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    pearson_r, pearson_p = stats.pearsonr(
        sub["pride_scint_rad"],
        sub["dsn_sigma_phi_rad"],
    )

    spearman_r, spearman_p = stats.spearmanr(
        sub["pride_scint_rad"],
        sub["dsn_sigma_phi_rad"],
    )

    return {
        "n": len(sub),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def build_log_transfer_df(df):
    out = df[
        (df["pride_scint_rad"] > 0) &
        (df["dsn_sigma_phi_rad"] > 0)
    ].copy()

    out["log_pride"] = np.log10(out["pride_scint_rad"])
    out["log_dsn"] = np.log10(out["dsn_sigma_phi_rad"])

    return out


def fit_log_transfer(df):
    fit_df = build_log_transfer_df(df)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fit_df["log_pride"],
        fit_df["log_dsn"],
    )

    return {
        "intercept": intercept,
        "alpha": slope,
        "r": r_value,
        "p": p_value,
        "alpha_err": std_err,
        "n": len(fit_df),
        "fit_df": fit_df,
    }


def fit_weighted_chi_transfer(df, frac_err_dsn=0.20):
    chi_df = build_log_transfer_df(df)
    chi_df["sigma_log_dsn"] = frac_err_dsn / np.log(10)

    def model(x, intercept, alpha):
        return intercept + alpha * x

    popt, pcov = curve_fit(
        model,
        chi_df["log_pride"].values,
        chi_df["log_dsn"].values,
        sigma=chi_df["sigma_log_dsn"].values,
        absolute_sigma=True,
    )

    intercept, alpha = popt
    intercept_err, alpha_err = np.sqrt(np.diag(pcov))

    model_log_dsn = model(
        chi_df["log_pride"].values,
        intercept,
        alpha,
    )

    residuals = chi_df["log_dsn"].values - model_log_dsn

    chi2 = np.sum((residuals / chi_df["sigma_log_dsn"].values) ** 2)
    dof = len(chi_df) - len(popt)
    reduced_chi2 = chi2 / dof

    observed_scatter = np.nanstd(residuals)
    measurement_scatter = np.nanmedian(chi_df["sigma_log_dsn"])

    intrinsic_scatter = np.sqrt(
        max(0, observed_scatter**2 - measurement_scatter**2)
    )

    return {
        "intercept": intercept,
        "alpha": alpha,
        "intercept_err": intercept_err,
        "alpha_err": alpha_err,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
        "observed_scatter_dex": observed_scatter,
        "measurement_scatter_dex": measurement_scatter,
        "intrinsic_scatter_dex": intrinsic_scatter,
        "intrinsic_scatter_factor": 10 ** intrinsic_scatter,
        "fit_df": chi_df,
    }


def add_transfer_residuals(df, intercept, alpha):
    resid_df = build_log_transfer_df(df)

    resid_df["log_dsn_model"] = (
        intercept + alpha * resid_df["log_pride"]
    )

    resid_df["transfer_residual"] = (
        resid_df["log_dsn"] - resid_df["log_dsn_model"]
    )

    resid_df["residual_factor"] = 10 ** resid_df["transfer_residual"]

    return resid_df


# ============================================================
# Multi-year DSN–PRIDE table construction
# ============================================================

def run_multiyear_dsn_pride_analysis(
    project_root,
    years,
    load_dsn_doppler_file,
    load_pride_scint_file,
    get_common_days,
    build_dsn_pride_binned_comparison,
    build_daily_dsn_pride_summary,
    compute_signal_correlations,
    compute_xcorr_summary,
    compute_sigma_phi_binned=None,
    data_root=None,
    bin_freq="20min",
    f_low_hz=3e-3,
    f_high_hz=0.1,
    detrend_poly_order=4,
    min_samples=16,
    max_valid_dsn_sigma=2.0,
    bin_minutes=20,
    min_bins_for_xcorr=4,
):
    project_root = Path(project_root)

    if data_root is None:
        data_root = project_root.parent
    else:
        data_root = Path(data_root)

    multi_summary = []
    multi_binned = []
    multi_qc = []
    multi_daily = []
    multi_xcorr = []
    all_dsn_bins = []

    for yr in years:
        print(f"\n========== YEAR {yr} ==========")

        dsn_file = data_root / "dataByYear" / f"data_{yr}.txt"
        pride_file = data_root / "scintdataByYear" / f"scint_{yr}.txt"

        if not dsn_file.exists():
            print("Missing DSN file:", dsn_file)
            continue

        if not pride_file.exists():
            print("Missing PRIDE file:", pride_file)
            continue

        try:
            dsn = load_dsn_doppler_file(dsn_file)
            pride = load_pride_scint_file(pride_file)

            print("DSN rows:", len(dsn))
            print("PRIDE rows:", len(pride))

            if compute_sigma_phi_binned is not None:
                dsn_only_bins = compute_sigma_phi_binned(
                    dsn,
                    bin_freq=bin_freq,
                    f_low_hz=f_low_hz,
                    f_high_hz=f_high_hz,
                    detrend_poly_order=detrend_poly_order,
                    min_samples=min_samples,
                )

                dsn_only_bins = (
                    dsn_only_bins
                    .reset_index(names="utc_time")
                    .assign(year=yr)
                )

                all_dsn_bins.append(dsn_only_bins)

            common = get_common_days(dsn, pride)

            print("Common observing days:", len(common))

            if len(common) == 0:
                continue

            binned = build_dsn_pride_binned_comparison(
                dsn,
                pride,
                common_days=common,
                bin_freq=bin_freq,
                f_low_hz=f_low_hz,
                f_high_hz=f_high_hz,
                detrend_poly_order=detrend_poly_order,
                min_samples=min_samples,
            ).sort_index()

            binned = add_daily_elongation_to_bins(
                binned=binned,
                project_root=project_root,
                data_root=data_root,
                year=yr,
            )

            binned["year"] = int(yr)

            qc = binned[
                (binned["dsn_sigma_phi_rad"] > 0) &
                (binned["dsn_sigma_phi_rad"] <= max_valid_dsn_sigma) &
                (binned["pride_scint_rad"] > 0)
            ].copy()

            qc["year"] = int(yr)
            qc["dsn_pride_ratio"] = (
                qc["dsn_sigma_phi_rad"] / qc["pride_scint_rad"]
            )
            qc["log10_dsn_pride_ratio"] = np.log10(qc["dsn_pride_ratio"])

            if len(qc) < 3:
                print("Too few QC bins.")
                continue

            daily = build_daily_dsn_pride_summary(qc)

            if "day" in daily.columns:
                daily["day"] = pd.to_datetime(daily["day"])
                daily = daily.sort_values("day").set_index("day")
            else:
                daily.index = pd.to_datetime(daily.index)
                daily = daily.sort_index()

            daily["year"] = int(yr)
            daily["dsn_pride_ratio"] = (
                daily["dsn_sigma_phi_rad"] / daily["pride_scint_rad"]
            )
            daily["log10_dsn_pride_ratio"] = np.log10(
                daily["dsn_pride_ratio"]
            )

            window_stats = compute_signal_correlations(qc)
            daily_stats = compute_signal_correlations(daily)

            window_spearman_r, window_spearman_p = safe_spearman(qc)
            daily_spearman_r, daily_spearman_p = safe_spearman(daily)

            xcorr = compute_xcorr_summary(
                qc,
                bin_minutes=bin_minutes,
                min_bins=min_bins_for_xcorr,
            )

            xcorr["year"] = int(yr)
            valid_xcorr = xcorr[xcorr["used_for_summary"]].copy()

            summary_row = {
                "year": int(yr),
                "common_days": len(common),
                "raw_bins": len(binned),
                "qc_bins": len(qc),
                "daily_rows": len(daily),

                "window_pearson_r": window_stats["pearson_r"],
                "window_pearson_p": window_stats["pearson_p"],
                "daily_pearson_r": daily_stats["pearson_r"],
                "daily_pearson_p": daily_stats["pearson_p"],

                "window_spearman_r": window_spearman_r,
                "window_spearman_p": window_spearman_p,
                "daily_spearman_r": daily_spearman_r,
                "daily_spearman_p": daily_spearman_p,

                "median_ratio": np.nanmedian(qc["dsn_pride_ratio"]),
                "ratio_q25": np.nanquantile(qc["dsn_pride_ratio"], 0.25),
                "ratio_q75": np.nanquantile(qc["dsn_pride_ratio"], 0.75),

                "frac_within_factor_2": np.mean(
                    (qc["dsn_pride_ratio"] >= 0.5) &
                    (qc["dsn_pride_ratio"] <= 2.0)
                ),

                "frac_dsn_stronger": np.mean(qc["dsn_pride_ratio"] > 1.0),
                "frac_dsn_weaker": np.mean(qc["dsn_pride_ratio"] < 1.0),

                "valid_xcorr_days": len(valid_xcorr),

                "median_best_lag_min": (
                    valid_xcorr["best_lag_minutes"].median()
                    if len(valid_xcorr) else np.nan
                ),

                "median_best_corr": (
                    valid_xcorr["best_corr"].median()
                    if len(valid_xcorr) else np.nan
                ),

                "median_zero_lag_corr": (
                    valid_xcorr["zero_lag_corr"].median()
                    if len(valid_xcorr) else np.nan
                ),

                "frac_near_zero_lag": (
                    (valid_xcorr["best_lag_minutes"].abs() <= bin_minutes).mean()
                    if len(valid_xcorr) else np.nan
                ),

                "frac_positive_zero_lag_corr": (
                    (valid_xcorr["zero_lag_corr"] > 0).mean()
                    if len(valid_xcorr) else np.nan
                ),
            }

            multi_summary.append(summary_row)
            multi_binned.append(binned)
            multi_qc.append(qc)
            multi_daily.append(daily)
            multi_xcorr.append(xcorr)

            print("QC bins:", len(qc))
            print("Daily Pearson r:", daily_stats["pearson_r"])
            print("Median DSN/PRIDE ratio:", np.nanmedian(qc["dsn_pride_ratio"]))

        except Exception as e:
            print(f"FAILED for {yr}: {e}")

    dsn_all_df = (
        pd.concat(all_dsn_bins, ignore_index=True)
        if all_dsn_bins else pd.DataFrame()
    )

    if not dsn_all_df.empty:
        dsn_all_df["utc_time"] = pd.to_datetime(dsn_all_df["utc_time"])
        dsn_all_df = dsn_all_df.sort_values("utc_time").reset_index(drop=True)

    return {
        "multi_summary_df": pd.DataFrame(multi_summary),
        "all_binned_df": (
            pd.concat(multi_binned, ignore_index=False)
            if multi_binned else pd.DataFrame()
        ),
        "all_qc_df": (
            pd.concat(multi_qc, ignore_index=False)
            if multi_qc else pd.DataFrame()
        ),
        "all_daily_df": (
            pd.concat(multi_daily, ignore_index=False)
            if multi_daily else pd.DataFrame()
        ),
        "all_xcorr_df": (
            pd.concat(multi_xcorr, ignore_index=True)
            if multi_xcorr else pd.DataFrame()
        ),
        "dsn_all_df": dsn_all_df,
    }


# ============================================================
# Transfer analysis
# ============================================================

def fit_sep_transfer(df):
    sep_df = df[
        (df["pride_scint_rad"] > 0) &
        (df["dsn_sigma_phi_rad"] > 0) &
        (df["elongation_deg"].notna())
    ].copy()

    sep_bins = [0, 5, 10, 20, 40, 90, 180]
    sep_labels = ["0–5", "5–10", "10–20", "20–40", "40–90", "90–180"]

    sep_df["sep_bin"] = pd.cut(
        sep_df["elongation_deg"],
        bins=sep_bins,
        labels=sep_labels,
        include_lowest=True,
    )

    sep_df["log_pride"] = np.log10(sep_df["pride_scint_rad"])
    sep_df["log_dsn"] = np.log10(sep_df["dsn_sigma_phi_rad"])

    rows = []

    for sep_bin, sub in sep_df.groupby("sep_bin", observed=True):
        sub = sub.dropna(subset=["log_pride", "log_dsn"])

        if len(sub) < 5:
            continue

        slope, intercept, r, p, stderr = stats.linregress(
            sub["log_pride"],
            sub["log_dsn"],
        )

        model = intercept + slope * sub["log_pride"]
        residuals = sub["log_dsn"] - model

        rows.append({
            "sep_bin": sep_bin,
            "n": len(sub),
            "median_sep": sub["elongation_deg"].median(),
            "alpha": slope,
            "alpha_err": stderr,
            "intercept": intercept,
            "r": r,
            "p": p,
            "scatter_dex": np.nanstd(residuals),
            "scatter_factor": 10 ** np.nanstd(residuals),
            "median_ratio": sub["dsn_pride_ratio"].median(),
        })

    return pd.DataFrame(rows)


def fit_state_transfer(df, q=0.75):
    state_df = build_log_transfer_df(df)
    threshold = state_df["pride_scint_rad"].quantile(q)

    state_df["state"] = np.where(
        state_df["pride_scint_rad"] >= threshold,
        "disturbed",
        "quiet/moderate",
    )

    rows = []

    for state, sub in state_df.groupby("state"):
        slope, intercept, r, p, stderr = stats.linregress(
            sub["log_pride"],
            sub["log_dsn"],
        )

        model = intercept + slope * sub["log_pride"]
        residuals = sub["log_dsn"] - model

        rows.append({
            "state": state,
            "n": len(sub),
            "alpha": slope,
            "alpha_err": stderr,
            "intercept": intercept,
            "r": r,
            "p": p,
            "scatter_dex": np.std(residuals),
            "scatter_factor": 10 ** np.std(residuals),
            "median_ratio": sub["dsn_pride_ratio"].median(),
        })

    return pd.DataFrame(rows), state_df


def summarise_residuals(resid_df):
    state_summary = None

    if "state" in resid_df.columns:
        state_summary = (
            resid_df
            .groupby("state")
            .agg(
                n=("transfer_residual", "size"),
                median_residual=("transfer_residual", "median"),
                q16_residual=("transfer_residual", lambda x: np.nanpercentile(x, 16)),
                q84_residual=("transfer_residual", lambda x: np.nanpercentile(x, 84)),
                scatter_dex=("transfer_residual", "std"),
                median_factor=("residual_factor", "median"),
            )
            .reset_index()
        )

    year_summary = (
        resid_df
        .groupby("year")
        .agg(
            n=("transfer_residual", "size"),
            median_residual=("transfer_residual", "median"),
            scatter_dex=("transfer_residual", "std"),
            median_factor=("residual_factor", "median"),
        )
        .reset_index()
    )

    return {
        "year_residual_summary": year_summary,
        "state_residual_summary": state_summary,
    }


def run_transfer_analysis(all_qc_df, frac_err_dsn=0.20):
    global_fit = fit_log_transfer(all_qc_df)

    chi_fit = fit_weighted_chi_transfer(
        all_qc_df,
        frac_err_dsn=frac_err_dsn,
    )

    resid_df = add_transfer_residuals(
        all_qc_df,
        intercept=chi_fit["intercept"],
        alpha=chi_fit["alpha"],
    )

    state_fit, state_df = fit_state_transfer(all_qc_df)

    q75 = resid_df["pride_scint_rad"].quantile(0.75)

    resid_df["state"] = np.where(
        resid_df["pride_scint_rad"] >= q75,
        "disturbed",
        "quiet/moderate",
    )

    sep_fit = fit_sep_transfer(all_qc_df)
    residual_summaries = summarise_residuals(resid_df)

    return {
        "global_fit": global_fit,
        "chi_fit": chi_fit,
        "sep_fit": sep_fit,
        "state_fit": state_fit,
        "state_df": state_df,
        "resid_df": resid_df,
        **residual_summaries,
    }


# ============================================================
# Plotting helpers
# ============================================================

def plot_long_term_overlap(dsn_all_df, combined_df):
    import matplotlib.dates as mdates

    if dsn_all_df is None or dsn_all_df.empty:
        print("No DSN-only dataframe available. Skipping long-term overlap plot.")
        return

    combined_plot = combined_df.copy()

    if "utc_time" not in combined_plot.columns:
        combined_plot = combined_plot.reset_index(names="utc_time")

    combined_plot["utc_time"] = pd.to_datetime(combined_plot["utc_time"])

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.scatter(
        dsn_all_df["utc_time"],
        dsn_all_df["dsn_sigma_phi_rad"],
        s=8,
        alpha=0.35,
        label="All DSN σφ",
    )

    ax.scatter(
        combined_plot["utc_time"],
        combined_plot["pride_scint_rad"],
        s=24,
        alpha=0.8,
        marker="^",
        label="PRIDE Scint_rad overlap",
    )

    ax.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2014-12-31"))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for yr in range(2010, 2015):
        ax.axvline(
            pd.Timestamp(f"{yr}-01-01"),
            color="black",
            alpha=0.12,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Phase scintillation (rad)")
    ax.set_title("All DSN phase scintillation with PRIDE overlap points (2010–2014)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_validation_summary(figure_df, xcorr_df, year="2010–2014", max_valid_sigma=2.0):
    stats_dict = simple_signal_correlations(figure_df)

    median_lag = xcorr_df["best_lag_minutes"].median()
    q25_lag = xcorr_df["best_lag_minutes"].quantile(0.25)
    q75_lag = xcorr_df["best_lag_minutes"].quantile(0.75)

    frac_near_zero = (xcorr_df["best_lag_minutes"].abs() <= 20).mean()
    frac_positive_zero = (xcorr_df["zero_lag_corr"] > 0).mean()

    example_day = figure_df.groupby("day").size().idxmax()
    example_sub = figure_df[figure_df["day"] == example_day].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.4]},
    )

    ax0, ax1, ax2 = axes

    x = figure_df["pride_scint_rad"]
    y = figure_df["dsn_sigma_phi_rad"]

    ax0.scatter(
        x,
        y,
        s=45,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )

    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())

    ax0.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        color="black",
        linewidth=1,
        label="1:1",
    )

    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 100)

    ax0.plot(
        xfit,
        m * xfit + b,
        color="red",
        linewidth=1.5,
        label="Linear fit",
    )

    ax0.set_xlabel("PRIDE Scint_rad (rad)")
    ax0.set_ylabel("DSN-derived σφ (rad)")
    ax0.set_title("(a) Amplitude comparison")
    ax0.legend(loc="best")

    ax0.text(
        0.04,
        0.96,
        f"N = {stats_dict['n']}\n"
        f"Pearson r = {stats_dict['pearson_r']:.2f}\n"
        f"p = {stats_dict['pearson_p']:.1e}\n"
        f"σφ ≤ {max_valid_sigma:g} rad",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.15, edgecolor="none"),
    )

    ax1.hist(
        xcorr_df["best_lag_minutes"].dropna(),
        bins=20,
        alpha=0.85,
        edgecolor="black",
    )

    ax1.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Zero lag",
    )

    ax1.set_xlabel("Best lag (minutes)")
    ax1.set_ylabel("Number of days")
    ax1.set_title("(b) Cross-correlation lag distribution")
    ax1.legend(loc="best")

    ax1.text(
        0.04,
        0.96,
        f"Median = {median_lag:.0f} min\n"
        f"IQR = {q25_lag:.0f} to {q75_lag:.0f} min\n"
        f"|lag| ≤ 20 min: {100*frac_near_zero:.0f}%\n"
        f"Zero-lag > 0: {100*frac_positive_zero:.0f}%",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.15, edgecolor="none"),
    )

    ax2.plot(
        example_sub.index,
        example_sub["dsn_sigma_phi_rad"],
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label="DSN-derived σφ",
    )

    ax2.plot(
        example_sub.index,
        example_sub["pride_scint_rad"],
        marker="^",
        linestyle="--",
        linewidth=1.8,
        markersize=5,
        label="PRIDE Scint_rad",
    )

    ax2.set_xlabel("UTC time")
    ax2.set_ylabel("Phase scintillation (rad)")
    ax2.set_title(f"(c) Example 20-min comparison: {example_day.date()}")
    ax2.legend(loc="best")

    fig.autofmt_xdate(rotation=30)

    plt.suptitle(
        f"DSN–PRIDE scintillation validation ({year})",
        fontsize=15,
        y=1.04,
    )

    plt.tight_layout()
    plt.show()


def plot_transfer_summary(results, fits):
    all_qc_df = results["all_qc_df"]
    chi_fit = fits["chi_fit"]
    fit_df = build_log_transfer_df(all_qc_df)

    fig, ax = plt.subplots(figsize=(7, 6))

    scatter = ax.scatter(
        fit_df["pride_scint_rad"],
        fit_df["dsn_sigma_phi_rad"],
        c=fit_df["year"],
        s=28,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.2,
    )

    xfit = np.logspace(
        np.log10(fit_df["pride_scint_rad"].min()),
        np.log10(fit_df["pride_scint_rad"].max()),
        200,
    )

    yfit = 10 ** chi_fit["intercept"] * xfit ** chi_fit["alpha"]

    ax.plot(
        xfit,
        yfit,
        color="red",
        linewidth=2.5,
        label=rf"$\alpha={chi_fit['alpha']:.2f}$",
    )

    ax.plot(
        xfit,
        xfit,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="1:1",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("PRIDE Scint_rad (rad)")
    ax.set_ylabel("DSN-derived σφ (rad)")
    ax.set_title("DSN–PRIDE transfer relation")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Year")

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_state_transfer(fits):
    state_df = fits["state_df"]
    state_fit = fits["state_fit"]

    fig, ax = plt.subplots(figsize=(7, 6))

    for state, sub in state_df.groupby("state"):
        ax.scatter(
            sub["pride_scint_rad"],
            sub["dsn_sigma_phi_rad"],
            s=28,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.2,
            label=state,
        )

        row = state_fit[state_fit["state"] == state].iloc[0]

        xfit = np.logspace(
            np.log10(sub["pride_scint_rad"].min()),
            np.log10(sub["pride_scint_rad"].max()),
            100,
        )

        yfit = 10 ** row["intercept"] * xfit ** row["alpha"]

        ax.plot(
            xfit,
            yfit,
            linewidth=2.5,
            label=rf"{state}: $\alpha={row['alpha']:.2f}$",
        )

    x_all = np.logspace(
        np.log10(state_df["pride_scint_rad"].min()),
        np.log10(state_df["pride_scint_rad"].max()),
        100,
    )

    ax.plot(
        x_all,
        x_all,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="1:1",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("PRIDE Scint_rad (rad)")
    ax.set_ylabel("DSN-derived σφ (rad)")
    ax.set_title("Quiet versus disturbed DSN–PRIDE transfer relation")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_residual_summary(fits):
    resid_df = fits["resid_df"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        resid_df["transfer_residual"],
        bins=40,
        alpha=0.75,
        edgecolor="black",
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel(r"Transfer residual $\Delta$ dex")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of DSN–PRIDE transfer residuals")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_year_residuals(fits):
    resid_df = fits["resid_df"]
    years = sorted(resid_df["year"].unique())

    data = [
        resid_df.loc[
            resid_df["year"] == year,
            "transfer_residual",
        ].dropna()
        for year in years
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        data,
        tick_labels=years,
        showfliers=False,
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(r"Transfer residual $\Delta$ dex")
    ax.set_title("Transfer residuals by year")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
    "figure_5_4_transfer_residuals.png",
    dpi=400,
    bbox_inches="tight"
    )
    plt.show()


def plot_fit_sensitivity(fits):
    from sklearn.linear_model import HuberRegressor

    compare_df = fits["resid_df"].copy()

    ordinary_alpha = fits["chi_fit"]["alpha"]
    ordinary_intercept = fits["chi_fit"]["intercept"]

    X = compare_df[["log_pride"]].values
    y = compare_df["log_dsn"].values

    huber = HuberRegressor(epsilon=1.5)
    huber.fit(X, y)

    robust_alpha = huber.coef_[0]
    robust_intercept = huber.intercept_

    filtered_df = compare_df[
        compare_df["transfer_residual"] > -0.4
    ].copy()

    filtered_fit = fit_weighted_chi_transfer(filtered_df)

    filtered_alpha = filtered_fit["alpha"]
    filtered_intercept = filtered_fit["intercept"]

    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        compare_df["pride_scint_rad"],
        compare_df["dsn_sigma_phi_rad"],
        c=compare_df["year"],
        cmap="viridis",
        s=55,
        alpha=0.70,
    )

    xgrid = np.logspace(
        np.log10(compare_df["pride_scint_rad"].min() * 0.8),
        np.log10(compare_df["pride_scint_rad"].max() * 1.2),
        300,
    )

    ax.plot(
        xgrid,
        xgrid,
        linestyle="--",
        color="black",
        linewidth=2,
        label="1:1",
    )

    ax.plot(
        xgrid,
        10 ** (ordinary_intercept + ordinary_alpha * np.log10(xgrid)),
        color="red",
        linewidth=3,
        label=f"Ordinary: α = {ordinary_alpha:.2f}",
    )

    ax.plot(
        xgrid,
        10 ** (robust_intercept + robust_alpha * np.log10(xgrid)),
        color="orange",
        linewidth=3,
        linestyle="-.",
        label=f"Robust: α = {robust_alpha:.2f}",
    )

    ax.plot(
        xgrid,
        10 ** (filtered_intercept + filtered_alpha * np.log10(xgrid)),
        color="limegreen",
        linewidth=3,
        linestyle=":",
        label=f"Filtered: α = {filtered_alpha:.2f}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("PRIDE Scint_rad (rad)")
    ax.set_ylabel("DSN-derived σφ (rad)")

    ax.set_title(
        "Sensitivity of the DSN–PRIDE transfer relation\n"
        "to outlier treatment"
    )

    ax.grid(True, which="both", alpha=0.3)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Year")

    ax.legend()

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "fit_type": ["ordinary", "robust", "filtered"],
        "alpha": [ordinary_alpha, robust_alpha, filtered_alpha],
        "intercept": [ordinary_intercept, robust_intercept, filtered_intercept],
    })