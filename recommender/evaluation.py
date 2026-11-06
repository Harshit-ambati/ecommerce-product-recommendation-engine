"""Full-catalog, fixed-cutoff ranking evaluation; no sampled negatives."""
import numpy as np
import pandas as pd
from .model import METHODS


def ranking_metrics(recommended, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        raise ValueError('Ranking metrics require a nonempty relevant set')
    hits = np.array([p in relevant for p in recommended[:k]], dtype=float)
    dcg = float((hits / np.log2(np.arange(len(hits)) + 2)).sum())
    ideal = float((1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum())
    return {'precision': float(hits.sum() / k), 'recall': float(hits.sum() / len(relevant)), 'ndcg': dcg / ideal}


def evaluate(model, future, k=10):
    if not future.empty and (future.timestamp < model.cutoff).any():
        raise ValueError('Evaluation events precede the training cutoff')
    positive = future[future.event_type.isin(['cart', 'purchase'])].copy()
    catalog = set(model.products.product_id)
    targets, audit = {}, {'positive_events': len(positive), 'unavailable_events': 0, 'seen_events': 0}
    for row in positive.itertuples():
        if row.product_id not in catalog:
            audit['unavailable_events'] += 1
        elif row.product_id in model.histories.get(row.user_id, {}):
            audit['seen_events'] += 1
        else:
            targets.setdefault(row.user_id, set()).add(row.product_id)
    audit['users_with_positive_events'] = int(positive.user_id.nunique())
    audit['eligible_users'] = len(targets)
    audit['users_without_eligible_targets'] = audit['users_with_positive_events'] - len(targets)
    if not targets:
        raise ValueError('No eligible held-out targets')
    cold_items = {p for p, i in model.item_index.items() if model.popularity[i] == 0}
    summaries, details = [], []
    for method in METHODS:
        all_recommended = set()
        cold_hits, cold_total = 0, 0
        for user, relevant in sorted(targets.items()):
            recommendations = model.recommend(user, k, method)
            ids = [r['product_id'] for r in recommendations]
            all_recommended.update(ids)
            cold_targets = relevant & cold_items
            cold_hits += len(set(ids) & cold_targets)
            cold_total += len(cold_targets)
            metrics = ranking_metrics(ids, relevant, k)
            details.append({'method': method, 'user_id': user,
                            'segment': 'warm' if user in model.histories else 'cold',
                            'relevant_count': len(relevant), **metrics})
        frame = pd.DataFrame([r for r in details if r['method'] == method])
        for segment in ['all', 'warm', 'cold']:
            subset = frame if segment == 'all' else frame[frame.segment == segment]
            summaries.append({'method': method, 'segment': segment, 'users': len(subset),
                              **{f'{m}@{k}': float(subset[m].mean()) if len(subset) else None for m in ['precision', 'recall', 'ndcg']},
                              'catalog_coverage': len(all_recommended) / len(catalog) if segment == 'all' else None,
                              'cold_item_recall': cold_hits / cold_total if cold_total and segment == 'all' else None,
                              'cold_item_targets': cold_total if segment == 'all' else None})
    return summaries, details, audit
