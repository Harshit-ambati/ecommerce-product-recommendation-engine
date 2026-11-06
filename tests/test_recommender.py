import numpy as np
import pandas as pd
import pytest
from recommender.data import generate, validate, VALIDATION, TEST, WEIGHTS
from recommender.model import Recommender, METHODS
from recommender.evaluation import ranking_metrics, evaluate


@pytest.fixture(scope='module')
def dataset():
    return generate(42)


@pytest.fixture(scope='module')
def model(dataset):
    return Recommender().fit(*dataset, VALIDATION)


def test_generator_repeatability(dataset):
    p, e = generate(42)
    pd.testing.assert_frame_equal(dataset[0], p)
    pd.testing.assert_frame_equal(dataset[1], e)
    assert len(e) >= 1000


def test_future_events_do_not_affect_training(dataset, model):
    products, events = dataset
    prior = events[events.timestamp < VALIDATION]
    clean = Recommender().fit(products, prior, VALIDATION)
    for method in METHODS:
        assert clean.recommend('U0000', method=method) == model.recommend('U0000', method=method)
    assert model.events.timestamp.max() < VALIDATION
    assert (model.products.available_at <= VALIDATION).all()
    assert 'P0340' not in model.item_index
    assert 'U0240' not in model.histories


def test_future_catalog_does_not_affect_features(dataset, model):
    products, events = dataset
    changed = products.copy()
    changed.loc[changed.available_at > VALIDATION, 'category'] = 'future_only_token'
    other = Recommender().fit(changed, events, VALIDATION)
    assert other.vectorizer.vocabulary_ == model.vectorizer.vocabulary_
    assert other.recommend('U0000') == model.recommend('U0000')


@pytest.mark.parametrize('method', METHODS)
def test_recommendations_exclude_seen_and_are_unique(model, method):
    rows = model.recommend('U0000', method=method)
    ids = [r['product_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 10
    assert not set(ids) & set(model.histories['U0000'])
    assert set(ids) <= set(model.item_index)
    assert all(np.isfinite(r['score']) and r['reason'] for r in rows)


def test_cold_user_falls_back(model):
    cold = model.recommend('new-user', method='content')
    popular = model.recommend('new-user', method='popularity')
    assert [r['product_id'] for r in cold] == [r['product_id'] for r in popular]
    assert all(r['fallback'] for r in cold)


def test_cold_item_content_can_score(model):
    new_index = model.item_index['P0320']
    assert model.popularity[new_index] == 0
    assert model.content_similarity[new_index].max() > 0
    assert model.collaborative_similarity[new_index].sum() == 0


def test_serialized_inference(model, tmp_path):
    import json
    artifact = tmp_path / 'model.pkl'
    model.save(artifact)
    restored = Recommender.load(artifact)
    assert restored.recommend('U0000') == model.recommend('U0000')
    assert json.loads(json.dumps(restored.recommend('new-user')))


def test_hand_calculated_metrics():
    metrics = ranking_metrics(['a', 'b', 'c'], {'a', 'c'}, k=3)
    assert metrics['precision'] == pytest.approx(2/3)
    assert metrics['recall'] == 1
    assert metrics['ndcg'] == pytest.approx(1.5 / (1 + 1/np.log2(3)))
    assert ranking_metrics([], {'a'})['ndcg'] == 0


def test_validation_rejects_bad_data(dataset):
    products, events = dataset
    bad = events.copy()
    bad.loc[0, 'product_id'] = 'missing'
    with pytest.raises(ValueError, match='Unknown product'):
        validate(products, bad)
    with pytest.raises(ValueError, match='Duplicate'):
        validate(products, pd.concat([events, events.head(1)]))
    bad = events.copy()
    bad.loc[0, 'timestamp'] = pd.Timestamp('2020-01-01')
    with pytest.raises(ValueError, match='before product'):
        validate(products, bad)


@pytest.mark.parametrize('k', [0, -1, 51, True, 1.2])
def test_invalid_k(model, k):
    with pytest.raises(ValueError):
        model.recommend('U0000', k)


def test_evaluation_audit_and_segments(dataset, model):
    events = dataset[1]
    future = events[(events.timestamp >= VALIDATION) & (events.timestamp < TEST)]
    summary, details, audit = evaluate(model, future)
    assert audit['eligible_users'] > 0
    assert any(r['segment'] == 'cold' and r['users'] > 0 for r in summary)
    assert all(0 <= r[m] <= 1 for r in details for m in ['precision', 'recall', 'ndcg'])
    with pytest.raises(ValueError, match='precede'):
        evaluate(model, events)
