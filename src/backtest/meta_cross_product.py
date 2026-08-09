"""Phase E1 — Cross-Product Evidence consensus gate (T-0301/T-0302).

Computes, per regime bucket, a weighted consensus across the three
components Analyst / News / ML and produces a **switch gate** that
``run_meta_decision`` consults before a strategy switch.

Key properties:

- Inputs are ``configured_weight_i`` (set by the user) and the
  *bucket-broken-down* ``conditioned_status_i`` together with the
  ``unconditional_status_i``. A component only counts if both
  unconditional PASS AND conditioned-for-the-bucket PASS hold
  (Codex R2.5).
- The gate acts ONLY as a switch gate on ``decision_bucket =
  current_regime_bucket`` (Codex R3.7). ``live_score`` remains
  untouched.
- Default threshold per profile: ``defensiv=0.7``, ``ausgewogen=0.5``,
  ``aggressiv=0.3``, ``custom=0.5`` (Codex R3.3).

Phase D stays bit-identical as long as ``cross_product_require=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_CROSS_PRODUCT_THRESHOLDS: Dict[str, float] = {
    "defensiv": 0.7,
    "ausgewogen": 0.5,
    "aggressiv": 0.3,
    "custom": 0.5,
}


REGIME_BUCKETS: tuple[str, ...] = (
    "normal",
    "news_stressed",
    "fragile",
    "stressed",
)


@dataclass(frozen=True)
class ComponentEvidence:
    """Consensus input of a single component (Analyst, News, or ML).

    Fields:
    - ``name``: ``"analyst"`` / ``"news"`` / ``"ml"``.
    - ``configured_weight``: weight value set by the user (NOT
      ``effective_weight``; Codex R2.5).
    - ``unconditional_status``: PASS/FAIL/STALE/MISSING from
      ``assess_<comp>_evidence``.
    - ``conditioned_status_by_bucket``: dict ``{bucket: status}`` from
      ``assess_conditioned_<comp>_evidence_status`` (Phase B+ / D).
    """

    name: str
    configured_weight: float
    unconditional_status: str
    conditioned_status_by_bucket: Mapping[str, str]


@dataclass(frozen=True)
class CrossProductGateResult:
    """Result of the cross-product switch gate."""

    status: str  # "pass" | "fail" | "missing" | "not_applicable"
    decision_bucket: Optional[str]
    consensus_per_bucket: Dict[str, float]
    threshold: float
    active_components: List[str]
    reasons: List[str]


def resolve_cross_product_threshold(
    profile: str,
    override: Optional[float] = None,
) -> float:
    """Resolves the consensus threshold value.

    A CLI override takes precedence over the profile default. For an
    unknown profile, we fall back to ``ausgewogen``.
    """

    if override is not None:
        return float(override)
    if profile in DEFAULT_CROSS_PRODUCT_THRESHOLDS:
        return DEFAULT_CROSS_PRODUCT_THRESHOLDS[profile]
    return DEFAULT_CROSS_PRODUCT_THRESHOLDS["ausgewogen"]


def bucket_consensus_map(
    components: Sequence[ComponentEvidence],
    buckets: Sequence[str] = REGIME_BUCKETS,
) -> Dict[str, float]:
    """Consensus weight summed per bucket (Codex R2.5).

    A component only counts if:
    (a) ``unconditional_status == "pass"``, and
    (b) ``conditioned_status_by_bucket.get(bucket) == "pass"``.

    Otherwise its contributed weight = 0.
    """

    out: Dict[str, float] = {bucket: 0.0 for bucket in buckets}
    for component in components:
        if component.unconditional_status != "pass":
            continue
        weight = float(component.configured_weight or 0.0)
        if weight <= 0:
            continue
        for bucket in buckets:
            if component.conditioned_status_by_bucket.get(bucket) == "pass":
                out[bucket] = out.get(bucket, 0.0) + weight
    return out


def active_components_list(components: Sequence[ComponentEvidence]) -> List[str]:
    """Components with ``configured_weight > 0``."""

    return sorted(
        c.name for c in components if float(c.configured_weight or 0.0) > 0
    )


def evaluate_cross_product_gate(
    components: Sequence[ComponentEvidence],
    *,
    current_regime_bucket: Optional[str],
    profile: str,
    threshold_override: Optional[float] = None,
    require: bool = False,
) -> CrossProductGateResult:
    """Pre-switch gate (Codex R3.7).

    Args:
        components: the three (Analyst, News, ML) component inputs.
        current_regime_bucket: from ``CandidateScore.regime_bucket`` of
            the *current* strategy row (NOT target — Codex R3.7).
        profile: evidence profile (`defensiv`/`ausgewogen`/`aggressiv`/`custom`).
        threshold_override: optional CLI override.
        require: if ``True``, the result is used as a gate;
            if ``False``, it is informational only
            (`status="not_applicable"`).

    Returns:
        ``CrossProductGateResult``. When ``require=False``, ``status``
        is always ``"not_applicable"``.
    """

    threshold = resolve_cross_product_threshold(profile, threshold_override)
    consensus = bucket_consensus_map(components)
    active = active_components_list(components)

    if not require:
        return CrossProductGateResult(
            status="not_applicable",
            decision_bucket=current_regime_bucket,
            consensus_per_bucket=consensus,
            threshold=threshold,
            active_components=active,
            reasons=[
                "cross_product_require=False; gate not evaluated"
            ],
        )

    # Missing handling (Codex R2.7): if no component is configured,
    # the consensus check is undefined -> `missing` instead of
    # `fail`.
    if not active:
        return CrossProductGateResult(
            status="missing",
            decision_bucket=current_regime_bucket,
            consensus_per_bucket=consensus,
            threshold=threshold,
            active_components=active,
            reasons=[
                "No active component with configured_weight > 0"
            ],
        )

    if not current_regime_bucket or current_regime_bucket == "insufficient_history":
        return CrossProductGateResult(
            status="missing",
            decision_bucket=current_regime_bucket,
            consensus_per_bucket=consensus,
            threshold=threshold,
            active_components=active,
            reasons=[
                "current_regime_bucket unavailable; cannot evaluate cross-product gate"
            ],
        )

    bucket_consensus = float(consensus.get(current_regime_bucket, 0.0))
    if bucket_consensus >= threshold:
        return CrossProductGateResult(
            status="pass",
            decision_bucket=current_regime_bucket,
            consensus_per_bucket=consensus,
            threshold=threshold,
            active_components=active,
            reasons=[
                f"consensus[{current_regime_bucket}]={bucket_consensus:.3f} "
                f">= threshold={threshold:.3f}"
            ],
        )
    return CrossProductGateResult(
        status="fail",
        decision_bucket=current_regime_bucket,
        consensus_per_bucket=consensus,
        threshold=threshold,
        active_components=active,
        reasons=[
            f"consensus[{current_regime_bucket}]={bucket_consensus:.3f} "
            f"< threshold={threshold:.3f}"
        ],
    )


def gate_switch_by_cross_product(
    result: CrossProductGateResult,
    *,
    target_strategy: str,
    current_strategy: str,
) -> Tuple[bool, List[str]]:
    """Applies the gate result to a concrete switch request.

    Returns ``(switch_allowed, reasons)``. For
    ``status="not_applicable"`` or ``target_strategy ==
    current_strategy``, ``switch_allowed=True`` (the gate does not
    block).
    """

    if target_strategy == current_strategy:
        return True, [
            "No switch requested (target == current); cross-product gate skipped"
        ]
    if result.status == "not_applicable":
        return True, list(result.reasons)
    if result.status == "pass":
        return True, list(result.reasons)
    return False, list(result.reasons)


__all__ = [
    "ComponentEvidence",
    "CrossProductGateResult",
    "DEFAULT_CROSS_PRODUCT_THRESHOLDS",
    "REGIME_BUCKETS",
    "active_components_list",
    "bucket_consensus_map",
    "evaluate_cross_product_gate",
    "gate_switch_by_cross_product",
    "resolve_cross_product_threshold",
]
