"""
Offline smoke test for the Streamlit dashboard.

Every network call is monkeypatched to fail, so the test proves that the
dashboard renders from the committed data (snapshot sample, market CSV, R
outputs) and surfaces visible warnings instead of crashing.
"""

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def _boom(*_, **__):
    raise ConnectionError("offline test")


def test_dashboard_renders_offline_with_fallbacks(monkeypatch):
    import calendar_layer
    import data_layer
    import flow_engine
    import positioning
    import validate

    monkeypatch.setattr(positioning, "snapshot", _boom)
    monkeypatch.setattr(positioning, "build_chain", _boom)
    monkeypatch.setattr(flow_engine, "live_option_flow", _boom)
    monkeypatch.setattr(data_layer, "fetch_ohlc", _boom)
    monkeypatch.setattr(calendar_layer, "next_event", _boom)
    monkeypatch.setattr(validate, "fetch_oracle", _boom)

    app = st_testing.AppTest.from_file("dashboard.py", default_timeout=120).run()
    assert not app.exception, app.exception
    warnings = " ".join(w.value for w in app.warning)
    assert "Live Deribit fetch failed" in warnings
    assert "last stored snapshot" in warnings
    assert "Oracle unavailable" in warnings
    captions = " ".join(c.value for c in app.caption)
    assert "Low-vol regime" in captions                  # Markov caption computed from JSON
    markdown = " ".join(m.value for m in app.markdown)
    assert "Out-of-sample verdict" in markdown
