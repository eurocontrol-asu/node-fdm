"""Explore mode-weight strategies on a realistic label distribution.

Drop-in standalone — run with `uv run python scripts/debug/explore_mode_weights.py`.
"""

from __future__ import annotations

import math

# Real counts from the user's last training run (mode_weights_computed log).
COUNTS: dict[str, int] = {
    "ALT_MACH": 491_269,
    "UNKVERT_UNK": 259_175,
    "ALT_CAS": 117_577,
    "GAMMA_MACH": 73_265,
    "VZ_CAS": 11_929,
    # 8 other labels exist but weren't in top/bottom5; we synthesise plausible
    # mid-range counts to keep n_labels = 13. Replace with real values if you
    # have them.
    "GAMMA_CAS_1": 60_000,
    "GAMMA_CAS_2": 45_000,
    "VZ_MACH": 30_000,
    "TURN_1": 25_000,
    "TURN_2": 20_000,
    "CLIMB_TRANS": 18_000,
    "DESC_TRANS": 16_000,
    "ACC_LEVEL": 14_000,
}


def cui_weights(counts: dict[str, int], beta: float) -> dict[str, float]:
    """Cui 2019 effective-number weights, normalised so weighted mean is 1."""
    raw: dict[str, float] = {}
    one_minus_beta = 1.0 - beta
    for label, c in counts.items():
        # In float64, beta**c can underflow to 0 for large c. That's fine — n_eff
        # saturates at 1/(1-beta), which is the source of the bug we observed.
        n_eff = (1.0 - beta**c) / one_minus_beta
        raw[label] = 1.0 / n_eff
    total = sum(counts.values())
    weighted_sum = sum(counts[k] * raw[k] for k in counts)
    norm = total / weighted_sum
    return {k: raw[k] * norm for k in counts}


def inverse_freq_weights(counts: dict[str, int]) -> dict[str, float]:
    """Inverse-frequency weights, normalised so weighted mean is 1."""
    raw = {k: 1.0 / c for k, c in counts.items()}
    total = sum(counts.values())
    weighted_sum = sum(counts[k] * raw[k] for k in counts)
    norm = total / weighted_sum
    return {k: raw[k] * norm for k in counts}


def sqrt_inverse_freq_weights(counts: dict[str, int]) -> dict[str, float]:
    """Inverse-sqrt-frequency (softer than 1/n, common in NLP)."""
    raw = {k: 1.0 / math.sqrt(c) for k, c in counts.items()}
    total = sum(counts.values())
    weighted_sum = sum(counts[k] * raw[k] for k in counts)
    norm = total / weighted_sum
    return {k: raw[k] * norm for k in counts}


def power_weights(counts: dict[str, int], alpha: float) -> dict[str, float]:
    """Generalised power weights w = 1/count**alpha, normalised."""
    raw = {k: 1.0 / (c**alpha) for k, c in counts.items()}
    total = sum(counts.values())
    weighted_sum = sum(counts[k] * raw[k] for k in counts)
    norm = total / weighted_sum
    return {k: raw[k] * norm for k in counts}


def summarise(name: str, w: dict[str, float], counts: dict[str, int]) -> None:
    vals = list(w.values())
    ratio = max(vals) / min(vals)
    cruise = w["ALT_MACH"]
    rare = w["VZ_CAS"]
    # Effective dataset re-weight: how much rare gets boosted vs cruise.
    boost = rare / cruise
    # Show top 5 and bottom 5 by weight.
    sorted_w = sorted(w.items(), key=lambda kv: kv[1], reverse=True)
    top5 = ", ".join(f"{k}={v:.2f}" for k, v in sorted_w[:3])
    bot5 = ", ".join(f"{k}={v:.2f}" for k, v in sorted_w[-3:])
    print(f"  {name:35s} | ratio={ratio:8.2f} | rare/cruise={boost:7.2f} |")
    print(f"  {'':35s} | top: {top5}")
    print(f"  {'':35s} | bot: {bot5}")
    print()


def main() -> None:
    print("=" * 80)
    print("INPUT DISTRIBUTION (13 labels)")
    print("=" * 80)
    total = sum(COUNTS.values())
    for k, v in sorted(COUNTS.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:20s} : {v:>8d}  ({100*v/total:5.2f}%)")
    print(f"  {'TOTAL':20s} : {total:>8d}")
    imbalance = max(COUNTS.values()) / min(COUNTS.values())
    print(f"  imbalance ratio max/min = {imbalance:.2f}")
    print()

    print("=" * 80)
    print("CUI 2019 — sweep beta")
    print("=" * 80)
    print("  beta close to 1 saturates: all n_eff -> 1/(1-beta), all weights equal.")
    print("  We need beta such that beta**count_min is still meaningfully > 0.")
    print()
    print(f"  count_min = {min(COUNTS.values())} -> need beta < ~"
          f"{math.exp(math.log(0.5) / min(COUNTS.values())):.6f} to keep beta**count_min ~ 0.5")
    print()
    for beta in [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99994, 0.99999, 0.999999]:
        w = cui_weights(COUNTS, beta)
        summarise(f"Cui beta={beta}", w, COUNTS)

    print("=" * 80)
    print("INVERSE-FREQUENCY VARIANTS")
    print("=" * 80)
    summarise("1/count (alpha=1)", inverse_freq_weights(COUNTS), COUNTS)
    summarise("1/sqrt(count) (alpha=0.5)", sqrt_inverse_freq_weights(COUNTS), COUNTS)

    print("=" * 80)
    print("POWER-LAW SWEEP — w = 1/count**alpha")
    print("=" * 80)
    print("  alpha=0   -> uniform (no reweighting)")
    print("  alpha=0.5 -> sqrt (softer)")
    print("  alpha=1   -> full inverse-frequency (aggressive)")
    print()
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0]:
        w = power_weights(COUNTS, alpha)
        summarise(f"power alpha={alpha}", w, COUNTS)

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
  Goal: boost rare classes 5-15x vs cruise. More than that risks gradient
  instability; less than that is too timid for an imbalance of ~49x.

  Tuned options to try (in order of conservatism):

  1. Cui beta=0.99994   -> mathematically calibrated for these counts.
     Currently the easiest fix: change auto_beta clipping to allow this.

  2. power alpha=0.5    -> sqrt inverse-frequency. Smooth, predictable,
     standard practice in NLP / detection. No edge cases at large counts.

  3. power alpha=0.25   -> milder still; minimal risk of overfitting to rare.

  AVOID:
  - Cui with default beta=0.99   -> degenerates to uniform (current bug).
  - power alpha=1.0              -> rare classes get >40x boost, unstable.
""")


if __name__ == "__main__":
    main()
