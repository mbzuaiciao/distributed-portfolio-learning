import numpy as np
import pytest

from hypergraph_portfolio.collaborative_lora import (
    combine_predictions,
    fit_simplex_weights,
    merge_lora_updates,
    subspace_similarity,
)


def test_equal_and_one_hot_mixture():
    p = np.array([[.1, .9], [.4, .6]])
    np.testing.assert_allclose(combine_predictions(p, [.5, .5]), [.5, .5])
    np.testing.assert_allclose(combine_predictions(p, [1, 0]), p[:, 0])


@pytest.mark.parametrize("p,w", [([[1.1, .2]], [.5, .5]), ([[np.nan]], [1]),
                                 ([[.2, .3]], [-1, 2]), ([[.2, .3]], [.2, .2]),
                                 ([[.2, .3]], [1]), ([], [])])
def test_invalid_mixture(p, w):
    with pytest.raises(ValueError):
        combine_predictions(p, w)


def test_learned_weights_development_optimum_and_duplicate_experts():
    p = np.array([[.1, .9], [.9, .1], [.2, .8], [.8, .2]])
    y = np.array([0, 1, 0, 1])
    w = fit_simplex_weights(p, y)
    assert w[0] > .999
    assert np.mean((p @ w - y)**2) < np.mean((p.mean(1)-y)**2)
    duplicate = np.column_stack([p[:, 0], p[:, 0]])
    np.testing.assert_allclose(fit_simplex_weights(duplicate, y), [.5, .5])
    np.testing.assert_array_equal(w, fit_simplex_weights(p, y))


def test_update_merge_avoids_cross_terms():
    factors = [(np.array([[2.]]), np.array([[3.]]), 1),
               (np.array([[5.]]), np.array([[7.]]), 2)]
    merged = merge_lora_updates(factors, [.25, .75])
    np.testing.assert_allclose(merged, [[54.]])
    assert merged[0, 0] != (.25*2+.75*5)*(.25*3+.75*7)


def test_subspace_rotation_invariance_and_orthogonality():
    b = np.eye(4)[:, :2]
    np.testing.assert_allclose(subspace_similarity(b, b @ np.array([[0, 1], [-1, 0]])), 1)
    assert subspace_similarity(b, np.eye(4)[:, 2:]) == 0
    assert subspace_similarity(b, b*0) == 0


def test_task_splits_are_disjoint_and_no_legacy_labels():
    import json

    from experiments.run_collaborative_lora_pilot import OUT, build_splits
    splits = build_splits(json.loads((OUT / "config.json").read_text()))
    ids = [r["event_id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids))
    assert all("label" not in r for rows in splits.values() for r in rows)


def test_colora_routes_task_transform_and_zero_initial_update():
    torch = pytest.importorskip("torch")
    from hypergraph_portfolio.collaborative_lora_layers import CollaborativeLoRALinear
    layer = CollaborativeLoRALinear(torch.nn.Linear(3, 2), rank=2, dropout=0, tasks=2)
    x = torch.ones(1, 3)
    torch.testing.assert_close(layer(x), layer.base_layer(x))
    with torch.no_grad():
        layer.lora_a.fill_(1)
        layer.lora_b.fill_(1)
        layer.lambdas[1].mul_(2)
    layer.task = 0
    delta0 = layer(x) - layer.base_layer(x)
    layer.task = 1
    torch.testing.assert_close(layer(x) - layer.base_layer(x), 2*delta0)


def test_label_loader_ignores_unrequested_outcomes_and_fixes_threshold(tmp_path, monkeypatch):
    from experiments import run_collaborative_lora_pilot as pilot
    root = tmp_path / "data/bayesian_lora"
    root.mkdir(parents=True)
    (root / "reconstructed_returns.csv").write_text(
        "episode_id,move_1d\ntrain_a,-0.06\ntrain_b,-0.04\ntest_hidden,NOT_INSPECTABLE\n")
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    labels = pilot.labels_for([{"event_id": "train_a"}, {"event_id": "train_b"}], -.05)
    np.testing.assert_array_equal(labels, [1, 0])


@pytest.mark.parametrize("factors", [
    [(np.ones((2, 3)), np.ones((2, 4)), 1)],
    [(np.ones((2, 2)), np.ones((2, 4)), np.inf)],
])
def test_invalid_merge_factors(factors):
    with pytest.raises(ValueError):
        merge_lora_updates(factors, [1])
