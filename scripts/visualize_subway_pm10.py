#!/usr/bin/env python3
"""Generate polished, reproducible figures for the subway congestion / PM10 analysis.

Inputs:
- data/raw/statjbnu1/data1.csv : subway congestion by station/direction/time
- data/raw/statjbnu1/data2.csv : station indoor air-quality measurements
- data/raw/statjbnu1/data3.csv : station list by district
- new_data4.csv                : line-level operation frequency summary

Outputs:
- assets/figures/*.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "statjbnu1"
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 240,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

PALETTE = {
    "navy": "#243B53",
    "blue": "#2F80ED",
    "cyan": "#2D9CDB",
    "teal": "#27AE60",
    "orange": "#F2994A",
    "red": "#EB5757",
    "purple": "#9B51E0",
    "gray": "#828282",
    "light": "#F2F6FA",
}


def normalize_line(x: str) -> int | None:
    if x in {"성수지선", "신정지선"}:
        return 2
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


def savefig(name: str) -> None:
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(path.relative_to(ROOT))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d1 = pd.read_csv(RAW / "data1.csv")
    d2 = pd.read_csv(RAW / "data2.csv")
    d3 = pd.read_csv(RAW / "data3.csv")
    freq = pd.read_csv(ROOT / "new_data4.csv")

    hour_cols = [c for c in d1.columns if ":" in c]
    d1[hour_cols] = d1[hour_cols].apply(pd.to_numeric, errors="coerce")
    d1["avg_congestion"] = d1[hour_cols].mean(axis=1)
    d1["line_num"] = d1["호선"].map(normalize_line)

    d2["line_num"] = pd.to_numeric(d2["호선"], errors="coerce").astype("float64")
    d2["station_clean"] = d2["역사명"].astype(str).str.replace(r"\d+$", "", regex=True).str.replace(" ", "", regex=False)

    station_cong = d1.groupby(["line_num", "역명"], as_index=False)["avg_congestion"].mean()
    station_cong["station_clean"] = station_cong["역명"].astype(str).str.replace(" ", "", regex=False)
    merged = d2.merge(
        station_cong[["line_num", "station_clean", "avg_congestion"]],
        on=["line_num", "station_clean"],
        how="left",
    )

    freq = freq.copy()
    freq["line_num"] = freq["호선"].astype(str).str.extract(r"(\d+)")
    freq["line_num"] = pd.to_numeric(freq["line_num"], errors="coerce")
    freq["freq"] = pd.to_numeric(freq["평일운행회수"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    # Use the subtotal row for line 2 and direct line rows for lines 1, 3--8.
    freq_line2 = freq[(freq["호선"].eq("2호선")) & (freq["구분3"].eq("소계"))]
    freq_direct = freq[(freq["호선"].astype(str).str.match(r"^[1-8]호선$")) & (freq["구분3"].eq(freq["호선"]))]
    freq_clean = pd.concat([freq_direct, freq_line2], ignore_index=True)
    freq_clean = freq_clean.drop_duplicates("line_num", keep="first")[["line_num", "freq"]]

    merged = merged.merge(freq_clean, on="line_num", how="left")
    return d1, d2, d3, merged, freq_clean


def fig_analysis_flow() -> None:
    """Numbered narrative flow (qualitative; no test statistics baked into the art)."""
    fig, ax = plt.subplots(figsize=(15, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1. Observation", "Indoor station PM10\nmeasured by station", PALETTE["blue"]),
        ("2. Initial hypothesis", "Rush-hour congestion\nmay raise PM10", PALETTE["blue"]),
        ("3. Cross-check", "Station-level patterns\nbreak the simple story", PALETTE["orange"]),
        ("4. Alternative factor", "Train operation\nfrequency (prior work)", PALETTE["purple"]),
        ("5. Statistical test", "Group comparison\n+ t-test", PALETTE["teal"]),
    ]
    n = len(steps)
    margin, gap = 0.012, 0.028
    bw = (1 - 2 * margin - gap * (n - 1)) / n
    y0, bh = 0.30, 0.46
    centers = []
    for i, (title, body, color) in enumerate(steps):
        x0 = margin + i * (bw + gap)
        cx = x0 + bw / 2
        centers.append((x0, cx))
        ax.add_patch(FancyBboxPatch(
            (x0, y0), bw, bh, boxstyle="round,pad=0.006,rounding_size=0.018",
            linewidth=2.4, edgecolor=color, facecolor="white", mutation_aspect=0.32))
        ax.add_patch(FancyBboxPatch(
            (x0, y0 + bh - 0.115), bw, 0.115, boxstyle="round,pad=0.006,rounding_size=0.018",
            linewidth=0, facecolor=color, alpha=0.12, mutation_aspect=0.32))
        ax.text(cx, y0 + bh - 0.058, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=color)
        ax.text(cx, y0 + bh * 0.40, body, ha="center", va="center",
                fontsize=10.5, color=PALETTE["navy"], linespacing=1.45)
    for i in range(n - 1):
        x_end = centers[i][0] + bw
        x_next = centers[i + 1][0]
        ax.add_patch(FancyArrowPatch(
            (x_end + 0.004, y0 + bh / 2), (x_next - 0.004, y0 + bh / 2),
            arrowstyle="-|>", mutation_scale=16, lw=2.2, color=PALETTE["gray"]))
    ax.text(0.5, 0.95, "From a Correlation-Looking Pattern to a Statistically Tested Factor",
            ha="center", va="center", fontsize=17, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.5, 0.075,
            "Goal: refuse a surface correlation and isolate train-operation frequency "
            "as an alternative explanation through hypothesis testing.",
            ha="center", va="center", fontsize=11, color=PALETTE["gray"])
    savefig("fig01_analysis_flow.png")


def fig_congestion_profile(d1: pd.DataFrame) -> None:
    hour_cols = [c for c in d1.columns if ":" in c]
    prof_all = d1[hour_cols].mean()
    prof_weekday = d1[d1["조사일자"].eq("평일")][hour_cols].mean()
    x = np.arange(len(hour_cols))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, prof_all.values, color=PALETTE["gray"], lw=2.2, label="All days")
    ax.plot(x, prof_weekday.values, color=PALETTE["blue"], lw=3.0, label="Weekday")
    ax.fill_between(x, prof_weekday.values, color=PALETTE["blue"], alpha=0.12)
    for start, end, label in [(5, 8, "AM peak"), (25, 28, "PM peak")]:
        ax.axvspan(start, end, color=PALETTE["orange"], alpha=0.14)
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.93, label, ha="center", color=PALETTE["orange"], fontsize=11)
    tick_idx = list(range(0, len(hour_cols), 3))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([hour_cols[i].replace(":00:00", ":00") for i in tick_idx], rotation=35, ha="right")
    ax.set_title("Average Subway Congestion by Time of Day")
    ax.set_xlabel("Time")
    ax.set_ylabel("Congestion (%)")
    ax.legend(frameon=False)
    sns.despine()
    savefig("fig02_congestion_hourly_profile.png")


def fig_pm10_by_line(d2: pd.DataFrame) -> None:
    d = d2.copy()
    order = sorted(d["line_num"].dropna().unique())
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.boxplot(data=d, x="line_num", y="미세먼지(PM10)", order=order, color="#D7E8FF", fliersize=2, linewidth=1.5, ax=ax)
    sns.stripplot(data=d, x="line_num", y="미세먼지(PM10)", order=order, color=PALETTE["blue"], alpha=0.5, size=4, jitter=0.22, ax=ax)
    line_mean = d.groupby("line_num")["미세먼지(PM10)"].mean()
    for i, line in enumerate(order):
        ax.scatter(i, line_mean.loc[line], s=90, color=PALETTE["red"], edgecolor="white", linewidth=1.2, zorder=4)
    ax.set_title("Station Indoor PM10 Distribution by Subway Line")
    ax.set_xlabel("Subway line")
    ax.set_ylabel("PM10")
    ax.text(0.99, 0.96, "red dot = line mean", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=PALETTE["gray"])
    sns.despine()
    savefig("fig03_pm10_by_line_distribution.png")


def fig_pm10_congestion_scatter(merged: pd.DataFrame) -> None:
    d = merged.dropna(subset=["avg_congestion", "미세먼지(PM10)", "line_num"]).copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(data=d, x="avg_congestion", y="미세먼지(PM10)", hue="line_num", palette="tab10", s=65, alpha=0.78, ax=ax)
    sns.regplot(data=d, x="avg_congestion", y="미세먼지(PM10)", scatter=False, color=PALETTE["navy"], line_kws={"lw": 2, "ls": "--"}, ax=ax)
    r, p = stats.pearsonr(d["avg_congestion"], d["미세먼지(PM10)"])
    ax.set_title("Station-level PM10 vs. Average Congestion")
    ax.set_xlabel("Average station congestion (%)")
    ax.set_ylabel("PM10")
    ax.text(0.03, 0.95, f"Pearson r = {r:.2f}\np-value = {p:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#D0D7DE"))
    ax.legend(title="Line", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine()
    savefig("fig04_pm10_vs_congestion_scatter.png")


def fig_frequency_overview(merged: pd.DataFrame) -> None:
    line = merged.dropna(subset=["freq", "미세먼지(PM10)"]).groupby("line_num", as_index=False).agg(
        pm10=("미세먼지(PM10)", "mean"),
        freq=("freq", "first"),
        stations=("미세먼지(PM10)", "size"),
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.scatterplot(data=line, x="freq", y="pm10", size="stations", hue="line_num", palette="tab10", sizes=(120, 520), legend=False, ax=ax)
    for _, row in line.iterrows():
        ax.text(row["freq"] + 8, row["pm10"], f"Line {int(row['line_num'])}", va="center", fontsize=10, color=PALETTE["navy"])
    ax.set_title("Line-level Train Frequency and Mean Station PM10")
    ax.set_xlabel("Weekday train operations")
    ax.set_ylabel("Mean station PM10")
    ax.text(0.02, 0.04, "Bubble size = number of measured stations", transform=ax.transAxes, fontsize=10, color=PALETTE["gray"])
    sns.despine()
    savefig("fig05_line_frequency_pm10_overview.png")


def fig_model_mse() -> None:
    d = pd.DataFrame({"Model": ["Linear Regression", "Random Forest"], "MSE": [72.105, 68.459]})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(d["Model"], d["MSE"], color=[PALETTE["gray"], PALETTE["blue"]], width=0.55)
    ax.set_title("Congestion Prediction Model Comparison")
    ax.set_ylabel("MSE, lower is better")
    ax.set_ylim(0, max(d["MSE"]) * 1.22)
    for bar, mse in zip(bars, d["MSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{mse:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.88, "Random Forest achieved lower error on the competition dataset.", transform=ax.transAxes,
            ha="center", fontsize=11, color=PALETTE["gray"])
    sns.despine(left=True)
    savefig("fig06_model_mse_comparison.png")


def fig_reported_ttest() -> None:
    d = pd.DataFrame({
        "Comparison": ["Train frequency\ndifference", "Congestion-only\ndifference"],
        "p_value": [0.05, 0.11],
        "label": ["p < 0.05", "p = 0.11"],
        "Significant": ["Significant", "Not significant"],
    })
    colors = [PALETTE["teal"], PALETTE["orange"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(d["Comparison"], d["p_value"], color=colors, width=0.55)
    ax.axhline(0.05, color=PALETTE["red"], lw=2, ls="--")
    ax.text(1.48, 0.052, "alpha = 0.05", color=PALETTE["red"], va="bottom", ha="right", fontsize=11)
    ax.set_title("Reported t-test Summary for PM10 Group Difference")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 0.14)
    for bar, label, sig in zip(bars, d["label"], d["Significant"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{label}\n{sig}", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.22, "Values are reproduced from the original project result summary.", transform=ax.transAxes,
            ha="center", fontsize=10, color=PALETTE["gray"])
    sns.despine()
    savefig("fig07_reported_ttest_summary.png")


def fig_group_comparison_design() -> None:
    """Recreate the comparison logic used in the final PDF at a higher visual quality.

    The original slides compare PM10 differences across groups separated by
    train-operation frequency and congestion. This figure visualizes the design,
    not a re-estimated test statistic.
    """
    groups = [
        ("Baseline", "Low frequency\nLow congestion", PALETTE["gray"]),
        ("Frequency effect", "High frequency\nLow congestion", PALETTE["teal"]),
        ("Combined high", "High frequency\nHigh congestion", PALETTE["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")
    xs = [0.18, 0.5, 0.82]
    for i, ((title, body, color), x) in enumerate(zip(groups, xs)):
        rect = plt.Rectangle((x - 0.115, 0.38), 0.23, 0.31, transform=ax.transAxes,
                             facecolor="white", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, 0.62, title, transform=ax.transAxes, ha="center", va="center",
                color=color, fontsize=13, fontweight="bold")
        ax.text(x, 0.49, body, transform=ax.transAxes, ha="center", va="center",
                color=PALETTE["navy"], fontsize=12)
    ax.annotate("PM10 difference", xy=(0.39, 0.535), xytext=(0.29, 0.535), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=2.2, color=PALETTE["teal"]),
                ha="center", va="bottom", color=PALETTE["teal"], fontsize=11)
    ax.annotate("PM10 difference", xy=(0.71, 0.535), xytext=(0.61, 0.535), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=2.2, color=PALETTE["blue"]),
                ha="center", va="bottom", color=PALETTE["blue"], fontsize=11)
    ax.text(0.5, 0.88, "Group Comparison Design: Frequency vs. Congestion", transform=ax.transAxes,
            ha="center", fontsize=20, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.5, 0.22,
            "Final presentation compared groups that differ in train frequency and/or congestion,\nthen used t-tests to check whether PM10 differences were statistically significant.",
            transform=ax.transAxes, ha="center", fontsize=12, color=PALETTE["gray"])
    savefig("fig08_group_comparison_design.png")


def main() -> None:
    d1, d2, d3, merged, freq_clean = load_data()
    fig_analysis_flow()
    fig_congestion_profile(d1)
    fig_pm10_by_line(d2)
    fig_pm10_congestion_scatter(merged)
    fig_frequency_overview(merged)
    fig_model_mse()
    fig_reported_ttest()
    fig_group_comparison_design()

    # Save derived table used by several figures for transparent reproducibility.
    derived = merged[["호선", "역사명", "미세먼지(PM10)", "이산화탄소(CO2)", "포름알데히드(HCHO)", "일산화탄소(CO)", "avg_congestion", "freq"]]
    derived.to_csv(OUT / "derived_station_pm10_congestion_frequency.csv", index=False)
    print((OUT / "derived_station_pm10_congestion_frequency.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
