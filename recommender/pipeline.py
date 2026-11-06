"""Run with python -m recommender.pipeline; all outputs are reproducible."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import pandas as pd
from .data import generate, save_data, START, VALIDATION, TEST, END
from .model import Recommender
from .evaluation import evaluate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run(root='.', seed=42):
    root = Path(root)
    report = root / 'reports'
    report.mkdir(parents=True, exist_ok=True)
    products, events = generate(seed)
    save_data(products, events, root / 'data' / 'raw')
    splits = {'train': events[events.timestamp < VALIDATION],
              'validation': events[(events.timestamp >= VALIDATION) & (events.timestamp < TEST)],
              'test': events[(events.timestamp >= TEST) & (events.timestamp < END)]}
    split_dir = root / 'data' / 'processed'
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rows.to_csv(split_dir / f'{name}.csv', index=False)
    validation_model = Recommender().fit(products, splits['train'], VALIDATION)
    val_summary, val_details, val_audit = evaluate(validation_model, splits['validation'])
    best = max((r for r in val_summary if r['segment'] == 'all'), key=lambda r: r['ndcg@10'])['method']
    # Freeze method choice before opening the test period; refit only on prior data.
    final = Recommender().fit(products, pd.concat([splits['train'], splits['validation']]), TEST)
    final.selected_method = best
    final.save(root / 'artifacts' / 'recommender.pkl')
    test_summary, test_details, test_audit = evaluate(final, splits['test'])
    for prefix, summary, details in [('validation', val_summary, val_details), ('test', test_summary, test_details)]:
        pd.DataFrame(summary).to_csv(report / f'{prefix}_metrics.csv', index=False)
        pd.DataFrame(details).to_csv(report / f'{prefix}_users.csv', index=False)
    daily = events.groupby(events.timestamp.dt.date).size().rename('events')
    daily.to_csv(report / 'daily_events.csv')
    events.merge(products[['product_id', 'category']], on='product_id').groupby('category').size().rename('events').to_csv(report / 'category_events.csv')
    eda = {'products': len(products), 'users': int(events.user_id.nunique()), 'events': len(events),
           'event_types': {k: int(v) for k, v in events.event_type.value_counts().items()},
           'missing_cells': int(products.isna().sum().sum() + events.isna().sum().sum()),
           'duplicate_event_ids': int(events.event_id.duplicated().sum()),
           'interaction_density': len(events[['user_id', 'product_id']].drop_duplicates()) / (len(products) * events.user_id.nunique()),
           'price_summary': {k: float(v) for k, v in products.price.describe().items()},
           'split_rows': {name: len(rows) for name, rows in splits.items()}}
    write_json(report / 'eda.json', eda)
    manifest = {'seed': seed, 'synthetic': True, 'selected_method': best,
                'selection_metric': 'validation NDCG@10', 'k': 10,
                'start': str(START), 'validation_cutoff': str(VALIDATION), 'test_cutoff': str(TEST), 'end': str(END),
                'validation_audit': val_audit, 'test_audit': test_audit,
                'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn']},
                'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / 'data' / 'raw').glob('*.csv'))}}
    write_json(report / 'run_manifest.json', manifest)
    sample = final.recommend(final.users[0])
    write_json(report / 'sample_recommendations.json', {'user_id': final.users[0], 'method': best, 'recommendations': sample})
    # Preserve both a successful case and the lowest-scoring warm-user case for review.
    selected_details = [r for r in test_details if r['method'] == best and r['segment'] == 'warm']
    cases = []
    for label, row in [('best', max(selected_details, key=lambda r: r['ndcg'])), ('worst', min(selected_details, key=lambda r: r['ndcg']))]:
        user = row['user_id']
        future = splits['test']
        relevant = sorted(set(future[(future.user_id == user) & future.event_type.isin(['cart', 'purchase'])].product_id)
                          & set(final.item_index) - set(final.histories.get(user, {})))
        cases.append({'case': label, **row, 'relevant_products': relevant, 'recommendations': final.recommend(user)})
    write_json(report / 'error_cases.json', cases)
    print(pd.DataFrame(test_summary).query('segment == "all"').to_string(index=False))
    print(f'Validation-selected method: {best}; artifact: {root / "artifacts" / "recommender.pkl"}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed)
