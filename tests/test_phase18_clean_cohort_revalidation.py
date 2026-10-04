"""Scientific provenance gates and hand-calculated cohort mechanics."""

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.run_phase9_1_cohort_return_audit import (
    build_audited_event_returns,
    group_weekly_cohorts,
)
from experiments.run_phase18_clean_cohort_revalidation import (
    META,
    OUT,
    ROOT,
    classify,
    clean_text,
    load_events,
    load_freeze,
    moving_blocks,
    records_for,
    resolve_historical_universe,
    sha256,
    signal_stat,
)
from hypergraph_portfolio.uncertainty import (
    build_cohort_stacked_weights,
    build_portfolio_ledger,
    compute_canonical_portfolio_metrics,
    simulate_cohort_stacked_portfolio,
)


@pytest.fixture(scope="module")
def freeze_data():
    cfg, commit = load_freeze()
    return cfg, commit, load_events(cfg)


def test_preregistration_committed_and_hashes_match(freeze_data):
    cfg, commit, _ = freeze_data
    assert commit
    assert cfg["portfolio"]["primary_horizons"] == [1, 4]
    assert cfg["model"]["epochs"] == 2
    assert cfg["model"]["weight_decay"] == 0.0
    assert cfg["portfolio"]["overlay"] is None
    assert cfg["model"]["mc_samples"] == 0
    assert sha256(META) == cfg["file_hashes"][str(META.relative_to(ROOT))]
    for path, digest in cfg["external_hashes"].items():
        assert sha256(path) == digest


def test_mutated_preregistration_rejected(monkeypatch, tmp_path):
    real = OUT / "preregistration.json"
    altered = tmp_path / "preregistration.json"
    altered.write_text(real.read_text().replace("0.7073", "0.9"))
    # Avoid modifying the actual frozen artifact; replace only the file read.
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: altered.read_text().encode() if p == real else original(p)
    )
    with pytest.raises(ValueError, match="differs from committed"):
        load_freeze()


def test_text_and_portfolio_formation_ignore_outcomes(freeze_data):
    _, _, (_, train, valid, oos, cutoff) = freeze_data
    df = oos.iloc[:30].copy()
    texts = df.apply(clean_text, axis=1)
    p = np.linspace(0.1, 0.3, len(df))
    a = group_weekly_cohorts(records_for(df, p, 0.2))
    df["move_1d"] = 0.95
    df["move_30d"] = -0.99
    df["label"] = 1 - df.label
    df["d_minus_1_close"] = 987654
    df["future_return"] = 999
    b = group_weekly_cohorts(records_for(df, p, 0.2))
    assert texts.equals(df.apply(clean_text, axis=1))
    for ea, eb in zip(a, b):
        np.testing.assert_allclose(ea["w_base"], eb["w_base"], atol=1e-14)
        np.testing.assert_allclose(ea["mu_hat"], eb["mu_hat"], atol=1e-14)
    assert train.date_dt.max() < valid.date_dt.min() < oos.date_dt.min()
    assert np.array_equal(oos.label, (oos.move_1d <= cutoff).astype(int))


def test_exact_native_identity_for_timestamp_ties(freeze_data):
    _, _, (events, *_) = freeze_data
    ids = resolve_historical_universe(events, "holdout")
    assert len(ids) == len(set(ids)) == 1728
    for seed in [123, 777]:
        assert ids == resolve_historical_universe(events, "holdout", seed)
    # Order of metadata tied rows cannot change identity reconstruction.
    assert ids == resolve_historical_universe(events.sample(frac=1, random_state=7), "holdout")


def toy_epochs():
    eps = []
    ret = {}
    for i in range(6):
        assets = [f"A{i}", f"B{i}"]
        ts = [f"2024-01-{1 + 7 * i:02d}"] * 2  # Simulator treats these as opaque keys.
        eps.append(
            {
                "week": str(i),
                "assets": assets,
                "timestamps": ts,
                "w_base": np.array([0.75, 0.25]),
                "real_r": np.array([0.02 * (i + 1), -0.01]),
                "mu_hat": np.zeros(2),
                "Q_market": np.eye(2),
            }
        )
        ret[(ts[0], assets[0])] = [0.02 * (i + 1), 0.03, -0.01, 0.04, 0.90]
        ret[(ts[0], assets[1])] = [-0.01, 0.02, 0.01, -0.02, -0.80]
    return eps, ret


def test_h1_reproduces_canonical_ledger_with_entry_aligned():
    eps, ret = toy_epochs()
    exposure = 0.7073
    sim = simulate_cohort_stacked_portfolio(eps, ret, 1, target_exposure=exposure)
    ledger = build_portfolio_ledger(eps, np.full(len(eps), exposure), "constant")
    net = np.array([r["net_return"] for r in ledger])
    net[0] -= exposure * 0.001
    np.testing.assert_allclose(sim["weekly_net_returns"], net, atol=1e-14)
    assert sim["weekly_traded_turnovers"][0] == exposure
    assert sim["weekly_traded_turnovers"][1] == pytest.approx(2 * exposure)


def test_h4_expiration_age_and_mtm_hand_check():
    eps, ret = toy_epochs()
    sim = simulate_cohort_stacked_portfolio(eps, ret, 4, target_exposure=1)
    weights = build_cohort_stacked_weights(eps, 4)
    assert "A0" not in weights[4]
    assert weights[3]["A0"] == pytest.approx(0.75 / 4)
    for t in range(6):
        start = max(0, t - 3)
        manual = sum(
            sum(
                ep["w_base"][i] * ret[(ep["timestamps"][i], a)][t - entry]
                for i, a in enumerate(ep["assets"])
            )
            for entry, ep in enumerate(eps[start : t + 1], start=start)
        ) / (t - start + 1)
        assert sim["weekly_gross_returns"][t] == pytest.approx(manual, abs=1e-14)
    # Expired cohort's enormous fifth-week return must never enter H=4.
    changed = copy.deepcopy(ret)
    for rs in changed.values():
        rs[4] = 10000
    assert (
        simulate_cohort_stacked_portfolio(eps, changed, 4)["weekly_gross_returns"]
        == sim["weekly_gross_returns"]
    )
    assert sim["weekly_base_turnovers"][4] == pytest.approx(0.5)


def test_audited_geometric_closure():
    df = pd.DataFrame([{"date": "2024-01-01", "symbol": "A", "company_name": "A", "move_1d": 0.10}])
    records = records_for(df, np.array([0.2]), 0.2)
    by_sym = {
        "A": [
            {"date": "2024-01-01", "move_30d": ".21", "d_minus_1_close": "100"},
            {"date": "2024-04-01", "move_30d": "0", "d_minus_1_close": "133.1"},
        ]
    }
    r = build_audited_event_returns(records, by_sym)[("2024-01-01", "A")]
    assert np.prod(1 + np.array(r[:4])) - 1 == pytest.approx(0.21, abs=1e-14)
    assert np.prod(1 + np.array(r)) - 1 == pytest.approx(0.331, abs=1e-14)
    assert r[1] == r[2] == r[3]


def test_canonical_utility_and_cost_decomposition():
    eps, ret = toy_epochs()
    h1 = simulate_cohort_stacked_portfolio(eps, ret, 1)
    h4 = simulate_cohort_stacked_portfolio(eps, ret, 4)
    n1, n4 = np.array(h1["weekly_net_returns"]), np.array(h4["weekly_net_returns"])
    dg = 0.7073 * np.mean(np.array(h4["weekly_gross_returns"]) - h1["weekly_gross_returns"])
    saved = 0.001 * np.mean(np.array(h1["weekly_traded_turnovers"]) - h4["weekly_traded_turnovers"])
    dv = 0.5 * (np.var(n1, ddof=1) - np.var(n4, ddof=1))
    assert h4["mean_net_utility"] - h1["mean_net_utility"] == pytest.approx(
        dg + saved + dv, abs=1e-14
    )
    assert compute_canonical_portfolio_metrics(n1)["mean_net_utility"] == pytest.approx(
        np.mean(n1) - 0.5 * np.var(n1, ddof=1)
    )


def test_constant_ranking_undefined_and_disjoint_quintiles():
    ids = np.array([str(i) for i in range(20)])
    assert np.isnan(signal_stat(np.zeros(20), np.arange(20), ids)).all()
    result = signal_stat(np.arange(20), np.arange(20), ids)
    assert result[0] == pytest.approx(1)
    np.testing.assert_allclose(result[2:], [1.5, 5.5, 9.5, 13.5, 17.5])


def test_block_resampling_and_frozen_categories():
    a = moving_blocks(69, 13, 1000, np.random.default_rng(42))
    assert a.shape == (1000, 69)
    np.testing.assert_array_equal(np.diff(a[:, :13], axis=1), np.ones((1000, 12)))
    assert classify(True, True, True, 0.001) == "A"
    assert classify(False, True, True, 0.001) == "B"
    assert classify(False, False, False, -0.001) == "C"
    assert classify(True, False, True, -0.001) == "D"
    assert classify(True, True, True, 0.001, False) == "D"


def test_freeze_precedes_completed_seed_artifacts():
    manifest = OUT / "training_manifest.json"
    if not manifest.exists():
        pytest.skip("Training has not yet completed")
    rows = json.loads(manifest.read_text())
    for row in rows:
        assert subprocess.check_output(
            [
                "git",
                "show",
                row["freeze_git_sha"]
                + ":artifacts/phase18_clean_cohort_revalidation/preregistration.json",
            ],
            cwd=ROOT,
        )
        assert row["recipe"]["checkpoint_selection"].startswith("fixed final epoch")
        assert row["checkpoint_sha256"] == sha256(
            OUT / f"clean_phase16_1_seed_{row['seed']}_state.pt"
        )
        assert row["predictions_sha256"] == sha256(
            OUT / f"clean_predictions_seed_{row['seed']}.csv"
        )
