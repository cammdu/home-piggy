"""NL policy rules: checks against the source documents and hand calculations."""
import csv
import hashlib
from importlib import resources

import numpy as np
import pytest

from homepiggy.affordability_nl import (
    BuyerProfile, FinancingCostTable, affordability_gap, annuity_factor, grossup_factor, load_policy,
)
from homepiggy.tax_nl import Box3Params, box3_tax

POLICY = load_policy(2026)
TABLE = FinancingCostTable.load(POLICY)
CSV_SHA256 = "5b1027f84ab2ba95f11d601d9af557f84a7d43c1bb943de437872c07adcd933d"


def test_table_file_unchanged_since_extraction():
    data = resources.files("homepiggy.data.nl").joinpath("financieringslast_2026_non_aow.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == CSV_SHA256


def test_table_structure():
    assert TABLE.pct.shape == (97, 12)
    assert TABLE.income_rows[0] == 0 and TABLE.income_rows[1] == 30000 and TABLE.income_rows[-1] == 125000
    assert np.all(np.diff(TABLE.pct, axis=1) >= 0)     # higher test rate -> higher allowed burden share


@pytest.mark.parametrize("income, rate, expected", [
    (57000, 0.0200, 0.185),    # Staatscourant 2025/36471 Tabel 1: 57.000 row, 1,501-2,000% column
    (60000, 0.0400, 0.226),    # 60.000 row, 3,501-4,000%
    (29999, 0.0150, 0.155),    # below 30.000 -> "–" row, <=1,500%
    (30000, 0.01501, 0.165),   # 1,501% -> second column
    (125000, 0.0700, 0.322),   # top row, >=6,501%
    (200000, 0.0650, 0.315),   # above top row -> top row (assumption); 6,500% is in 6,001-6,500
    (79500, 0.0425, 0.262),    # 79.000 row
])
def test_table_lookup_cells(income, rate, expected):
    assert TABLE.lookup(np.array([income]), np.array([rate]))[0] == pytest.approx(expected)


def test_annuity_factor():
    assert annuity_factor(np.array([0.0]), 360)[0] == 360
    assert annuity_factor(np.array([0.045 / 12]), 360)[0] == pytest.approx(197.3612, abs=1e-3)


@pytest.mark.parametrize("rate, f", [(0.0200, 1.05), (0.02001, 1.10), (0.0350, 1.20), (0.0451, 1.30), (0.0601, 1.40)])
def test_student_grossup(rate, f):
    assert grossup_factor(POLICY, np.array([rate]))[0] == f


def _gap(price=300000.0, income=60000.0, rate=0.04, instal=0.0, age=28.0, **buyer):
    b = BuyerProfile(**{"use_nhg": False, **buyer})
    return affordability_gap(np.array([price]), np.array([income]), np.array([rate]), np.array([instal]),
                             age, np.array([1.0]), np.array([1.0]), b, POLICY, TABLE)


def test_worked_example_single_buyer():
    r = _gap()
    loan = 60000 * 0.226 / 12 * annuity_factor(np.array([0.04 / 12]), 360)[0] + 17000
    assert r.income_loan[0] == pytest.approx(loan, rel=1e-12)
    assert r.transfer_tax[0] == 0.0                                    # starter, price below 555k
    assert r.required_deposit[0] == pytest.approx(max(300000 - min(loan, 300000), 0))


def test_short_fixed_period_uses_5pct_minimum():
    r = _gap(rate=0.035, fixed_period_years=5)
    assert r.test_rate[0] == 0.05


def test_student_debt_reduces_loan():
    a, b = _gap(), _gap(instal=200.0)
    diff = a.income_loan[0] - b.income_loan[0]
    assert diff == pytest.approx(200 * 1.20 * annuity_factor(np.array([0.04 / 12]), 360)[0], rel=1e-12)


def test_starters_exemption_rules():
    assert _gap(age=34.9).transfer_tax[0] == 0.0
    assert _gap(age=35.0).transfer_tax[0] == pytest.approx(0.02 * 300000)
    assert _gap(price=560000).transfer_tax[0] == pytest.approx(0.02 * 560000)   # above limit: no exemption at all
    assert _gap(first_time_buyer=False).transfer_tax[0] == pytest.approx(6000)


def test_ltv_cap_and_couple():
    r = _gap(price=100000, income=125000)
    assert r.max_loan[0] == 100000 and r.required_deposit[0] == 0.0
    assert _gap(single=False).income_loan[0] == pytest.approx(_gap().income_loan[0] - 17000)


def test_nhg_premium():
    r = _gap(use_nhg=True)
    assert r.nhg_premium[0] == pytest.approx(0.004 * r.max_loan[0])


def test_box3_hand_example():
    p = Box3Params.from_policy(POLICY)
    tax = box3_tax(np.array([0.0]), np.array([100000.0]), p)[0]
    # G=100,000; deemed 6% -> p=6%; B = 100,000 - 59,357 = 40,643; voordeel 2,438.58; tax 36% = 877.89
    assert tax == pytest.approx(0.36 * 0.06 * (100000 - 59357), rel=1e-12)
    assert box3_tax(np.array([0.0]), np.array([50000.0]), p)[0] == 0.0


def test_box3_mixed_categories_and_other_assets():
    p = Box3Params.from_policy(POLICY, other_box3_assets=50000.0)
    tax = box3_tax(np.array([20000.0]), np.array([30000.0]), p)[0]
    G = 100000.0
    pct = (20000 * 0.0128 + 80000 * 0.06) / G
    total = 0.36 * pct * (G - 59357)
    assert tax == pytest.approx(total * 0.5, rel=1e-12)              # fund is half of box 3 assets


def test_policy_entries_all_have_source_and_verification():
    def walk(d, path=""):
        if isinstance(d, dict):
            if "verification" in d:
                assert d["verification"] in ("primary", "secondary", "assumption"), path
                assert any(k.startswith("source") for k in d) or d["verification"] == "assumption", path
            for k, v in d.items():
                walk(v, f"{path}.{k}")
    walk(POLICY)
