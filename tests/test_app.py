"""Smoke test of the Streamlit app with Streamlit's own AppTest harness (small run, single process)."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_runs_end_to_end(monkeypatch):
    import homepiggy.analysis as an
    orig = an.band_study
    monkeypatch.setattr(an, "band_study", lambda *a, **k: orig(*a, **{**k, "n_jobs": 1}))
    at = st_testing.AppTest.from_file("../streamlit_app/app.py", default_timeout=600)
    at.run()
    # shrink the run: one method, 25 re-calibrations, 250 paths, one strategy
    at.radio[0].set_value("Repo classifier (primary)")
    at.select_slider[0].set_value(25)
    at.select_slider[1].set_value(250)
    at.multiselect[0].set_value(["Agreed step glidepath"])
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert any("P(reach own target)" in str(df.value.columns.tolist()) for df in at.dataframe)
