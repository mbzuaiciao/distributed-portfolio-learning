"""Descriptive reporting only; consume completed frozen Phase18 outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase18_clean_cohort_revalidation"


def main():
    signals = pd.read_csv(OUT / "clean_signal_decay.csv")
    old = pd.read_csv(
        ROOT / "artifacts/phase9_1_cohort_return_audit/signal_decay_normalized.csv"
    )
    cumulative = signals[
        (signals.system == "clean_lora") & (signals.kind == "cumulative")
    ].copy()
    comparison = cumulative.merge(old, left_on="horizon", right_on="horizon_weeks")
    comparison["benchmark_status"] = (
        "historical/compromised-lineage; exact original leakage unverified; not a control"
    )
    comparison.to_csv(OUT / "signal_comparison_phase9_1.csv", index=False)
    age = signals[(signals.system == "clean_lora") & (signals.kind == "age_week")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, data, label in [
        (axes[0], cumulative, "Cumulative return"),
        (axes[1], age, "Individual reconstructed week"),
    ]:
        ax.plot(data.horizon, data.rank_ic, "o-", color="#176b87", label="Clean fixed recipe")
        ax.fill_between(
            data.horizon, data.ic_ci_lower, data.ic_ci_upper, color="#176b87", alpha=0.15
        )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set(xlabel="Horizon / age (weeks)", ylabel="Rank IC", title=label)
        ax.grid(alpha=0.15)
    axes[0].plot(
        old.horizon_weeks, old.mean_rank_ic, "s--", color="#9e6a42",
        label="Historical Phase 9.1 benchmark",
    )
    axes[0].legend(fontsize=8)
    axes[1].axvspan(2, 4, color="#e3bf48", alpha=0.12)
    fig.suptitle(
        "Phase 18 clean signal decay\n"
        "Interpolated event endpoints; shading: paired calendar-block 95% intervals",
        fontsize=11,
    )
    fig.savefig(OUT / "clean_signal_decay.png", dpi=180)
    fig.savefig(OUT / "clean_signal_decay.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
