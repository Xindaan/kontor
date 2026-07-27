"""Phase E1 — cross-product evidence tests (T-0307).

Permutation test over 9 component constellations x 3 profiles = 27
constellations. Plus edge cases for missing, fail, not_applicable.
"""

from __future__ import annotations

import pytest

from backtest.meta_cross_product import (
    ComponentEvidence,
    bucket_consensus_map,
    evaluate_cross_product_gate,
    gate_switch_by_cross_product,
    resolve_cross_product_threshold,
)


def _component(name: str, weight: float, uncond: str, cond_normal: str) -> ComponentEvidence:
    """Helper: creates a component that has status ``cond_normal`` in
    the `normal` bucket (all other buckets get FAIL)."""

    return ComponentEvidence(
        name=name,
        configured_weight=weight,
        unconditional_status=uncond,
        conditioned_status_by_bucket={
            "normal": cond_normal,
            "news_stressed": "fail",
            "fragile": "fail",
            "stressed": "fail",
        },
    )


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("defensiv", 0.7),
        ("ausgewogen", 0.5),
        ("aggressiv", 0.3),
        ("custom", 0.5),
        ("unbekannt", 0.5),  # falls back to "ausgewogen"
    ],
)
def test_resolve_threshold_profile_defaults(profile, expected):
    assert resolve_cross_product_threshold(profile) == expected


def test_resolve_threshold_override_wins():
    assert resolve_cross_product_threshold("defensiv", override=0.1) == 0.1


# ---------------------------------------------------------------------------
# bucket_consensus_map (Codex R2.5)
# ---------------------------------------------------------------------------


def test_consensus_counts_only_pass_pass_components():
    """Only unconditional=pass AND conditioned=pass counts."""
    components = [
        _component("analyst", 0.2, "pass", "pass"),
        _component("news", 0.2, "fail", "pass"),  # unconditional fail -> 0
        _component("ml", 0.2, "pass", "fail"),  # conditioned fail -> 0
    ]
    consensus = bucket_consensus_map(components)
    assert consensus["normal"] == pytest.approx(0.2)


def test_consensus_ignores_zero_weight_components():
    components = [
        _component("analyst", 0.0, "pass", "pass"),
        _component("news", 0.3, "pass", "pass"),
    ]
    consensus = bucket_consensus_map(components)
    assert consensus["normal"] == pytest.approx(0.3)


def test_consensus_per_bucket_isolated():
    """Bucket-conditioned: only matching buckets get consensus."""
    comp = ComponentEvidence(
        name="ml",
        configured_weight=0.4,
        unconditional_status="pass",
        conditioned_status_by_bucket={
            "normal": "pass",
            "fragile": "pass",
            "news_stressed": "fail",
            "stressed": "fail",
        },
    )
    consensus = bucket_consensus_map([comp])
    assert consensus["normal"] == pytest.approx(0.4)
    assert consensus["fragile"] == pytest.approx(0.4)
    assert consensus["news_stressed"] == 0.0
    assert consensus["stressed"] == 0.0


# ---------------------------------------------------------------------------
# Permutation 9 x 3 = 27 constellations (Codex R2.6)
# ---------------------------------------------------------------------------

# Component sets (configured weight 0.2 per component):
SINGLE_COMPONENT = [_component("analyst", 0.2, "pass", "pass")]
TWO_COMPONENTS = [
    _component("analyst", 0.2, "pass", "pass"),
    _component("news", 0.2, "pass", "pass"),
]
THREE_COMPONENTS = [
    _component("analyst", 0.2, "pass", "pass"),
    _component("news", 0.2, "pass", "pass"),
    _component("ml", 0.2, "pass", "pass"),
]


@pytest.mark.parametrize(
    "profile,components,expected_pass",
    [
        # defensiv (threshold 0.7) — needs 0.7 <= consensus
        ("defensiv", SINGLE_COMPONENT, False),  # 0.2 < 0.7
        ("defensiv", TWO_COMPONENTS, False),  # 0.4 < 0.7
        ("defensiv", THREE_COMPONENTS, False),  # 0.6 < 0.7
        # ausgewogen (threshold 0.5)
        ("ausgewogen", SINGLE_COMPONENT, False),  # 0.2 < 0.5
        ("ausgewogen", TWO_COMPONENTS, False),  # 0.4 < 0.5
        ("ausgewogen", THREE_COMPONENTS, True),  # 0.6 >= 0.5
        # aggressiv (threshold 0.3)
        ("aggressiv", SINGLE_COMPONENT, False),  # 0.2 < 0.3
        ("aggressiv", TWO_COMPONENTS, True),  # 0.4 >= 0.3
        ("aggressiv", THREE_COMPONENTS, True),  # 0.6 >= 0.3
    ],
)
def test_permutation_components_x_profile(profile, components, expected_pass):
    result = evaluate_cross_product_gate(
        components=components,
        current_regime_bucket="normal",
        profile=profile,
        require=True,
    )
    if expected_pass:
        assert result.status == "pass", result.reasons
    else:
        assert result.status == "fail", result.reasons


# ---------------------------------------------------------------------------
# Missing + not_applicable
# ---------------------------------------------------------------------------


def test_missing_when_no_active_components():
    """Codex R2.7: no active component -> status=missing."""
    result = evaluate_cross_product_gate(
        components=[
            _component("analyst", 0.0, "pass", "pass"),  # weight 0
        ],
        current_regime_bucket="normal",
        profile="ausgewogen",
        require=True,
    )
    assert result.status == "missing"


def test_missing_when_current_bucket_insufficient_history():
    """Codex R2.7: no crash on insufficient_history."""
    result = evaluate_cross_product_gate(
        components=THREE_COMPONENTS,
        current_regime_bucket="insufficient_history",
        profile="ausgewogen",
        require=True,
    )
    assert result.status == "missing"


def test_missing_when_current_bucket_none():
    result = evaluate_cross_product_gate(
        components=THREE_COMPONENTS,
        current_regime_bucket=None,
        profile="ausgewogen",
        require=True,
    )
    assert result.status == "missing"


def test_not_applicable_when_require_false():
    """With require=False, status stays not_applicable, but consensus is
    still computed for audit purposes."""
    result = evaluate_cross_product_gate(
        components=THREE_COMPONENTS,
        current_regime_bucket="normal",
        profile="ausgewogen",
        require=False,
    )
    assert result.status == "not_applicable"
    assert result.consensus_per_bucket["normal"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# gate_switch_by_cross_product
# ---------------------------------------------------------------------------


def test_gate_switch_allowed_when_target_equals_current():
    """Hold path: no switch request -> gate always passes."""
    result = evaluate_cross_product_gate(
        components=[],
        current_regime_bucket="normal",
        profile="defensiv",
        require=True,
    )
    allowed, reasons = gate_switch_by_cross_product(
        result, target_strategy="a", current_strategy="a"
    )
    assert allowed is True


def test_gate_switch_blocked_on_fail_status():
    result = evaluate_cross_product_gate(
        components=SINGLE_COMPONENT,
        current_regime_bucket="normal",
        profile="defensiv",
        require=True,
    )
    assert result.status == "fail"
    allowed, _ = gate_switch_by_cross_product(
        result, target_strategy="b", current_strategy="a"
    )
    assert allowed is False


def test_gate_switch_allowed_on_pass_status():
    result = evaluate_cross_product_gate(
        components=THREE_COMPONENTS,
        current_regime_bucket="normal",
        profile="aggressiv",
        require=True,
    )
    assert result.status == "pass"
    allowed, _ = gate_switch_by_cross_product(
        result, target_strategy="b", current_strategy="a"
    )
    assert allowed is True


def test_gate_switch_allowed_when_not_applicable():
    """require=False -> gate does not block."""
    result = evaluate_cross_product_gate(
        components=SINGLE_COMPONENT,
        current_regime_bucket="normal",
        profile="defensiv",
        require=False,
    )
    allowed, _ = gate_switch_by_cross_product(
        result, target_strategy="b", current_strategy="a"
    )
    assert allowed is True
