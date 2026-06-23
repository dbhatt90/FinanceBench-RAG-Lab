"""Tests for the CRAG conditional edge — verifies the threshold and enable flag
are read from state (the bug being fixed: they were hardcoded in the graph)."""

from rag_hub.routing.graph import _crag_selector


def test_low_confidence_routes_to_web_fallback():
    state = {"crag_confidence": 0.3, "crag_threshold": 0.5, "web_fallback_enabled": True}
    assert _crag_selector(state) == "web_fallback"


def test_high_confidence_routes_to_generate():
    state = {"crag_confidence": 0.8, "crag_threshold": 0.5, "web_fallback_enabled": True}
    assert _crag_selector(state) == "generate"


def test_state_threshold_overrides_default():
    # confidence 0.6 is below a high configured threshold → fallback, even though
    # it would pass the module default of 0.5. This is exactly the bug being fixed.
    state = {"crag_confidence": 0.6, "crag_threshold": 0.95, "web_fallback_enabled": True}
    assert _crag_selector(state) == "web_fallback"


def test_disabled_fallback_always_generates():
    state = {"crag_confidence": 0.1, "crag_threshold": 0.5, "web_fallback_enabled": False}
    assert _crag_selector(state) == "generate"


def test_already_used_fallback_does_not_loop():
    state = {
        "crag_confidence": 0.1,
        "crag_threshold": 0.5,
        "web_fallback_enabled": True,
        "used_fallback": True,
    }
    assert _crag_selector(state) == "generate"
