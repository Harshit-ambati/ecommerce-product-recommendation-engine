"""Synthetic inputs and strict validation; no real customer data."""
from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp('2025-01-01')
VALIDATION = START + pd.Timedelta(days=120)
TEST = START + pd.Timedelta(days=150)
END = START + pd.Timedelta(days=180)
WEIGHTS = {'view': 1.0, 'cart': 3.0, 'purchase': 5.0}
CATEGORIES = ['Audio', 'Outdoors', 'Kitchen', 'Fitness', 'Office', 'Travel']


def generate(seed=42):
    """Simulate 180 days. Latent preferences never become model features."""
    rng = np.random.default_rng(seed)
    n_items, n_users = 360, 300
    categories = np.arange(n_items) % len(CATEGORIES)
    styles = rng.integers(0, 4, n_items)
    brands = rng.integers(0, 8, n_items)
    launches = np.zeros(n_items, dtype=int)
    launches[300:320] = 100
    launches[320:340] = 120
    launches[340:] = 150
    products = pd.DataFrame({
        'product_id': [f'P{i:04d}' for i in range(n_items)],
        'name': [f'{["Essential", "Studio", "Active", "Compact"][styles[i]]} {CATEGORIES[categories[i]]} {i+1:03d}' for i in range(n_items)],
        'category': [CATEGORIES[c] for c in categories],
        'style': [['essential', 'studio', 'active', 'compact'][s] for s in styles],
        'brand': [f'Brand{b}' for b in brands],
        'price': np.round(rng.uniform(15, 220, n_items), 2),
        'available_at': [START + pd.Timedelta(days=int(d)) for d in launches],
    })
    tastes = rng.dirichlet(np.full(6, .35), n_users)
    favorite_style = rng.integers(0, 4, n_users)
    budgets = rng.uniform(35, 180, n_users)
    appeal = rng.lognormal(0, .7, n_items)
    activity = rng.lognormal(0, .55, n_users)
    joins = np.zeros(n_users, dtype=int)
    joins[240:270] = 120
    joins[270:] = 150
    rows = []
    for day in range(180):
        for u in np.flatnonzero(joins <= day):
            if rng.random() > min(.75, .19 * activity[u]):
                continue
            available = np.flatnonzero(launches <= day)
            preference = tastes[u, categories[available]]
            price_fit = np.exp(-np.abs(products.price.to_numpy()[available] - budgets[u]) / 120)
            affinity = (.08 + preference) * (1 + .6 * (styles[available] == favorite_style[u])) * price_fit
            # Mild seasonality and drift; stochastic events and repeat visits are retained.
            seasonal = 1 + .3 * np.sin(day / 20 + categories[available])
            probs = affinity * appeal[available] * seasonal
            probs /= probs.sum()
            for _ in range(int(rng.integers(1, 4))):
                j = int(rng.choice(available, p=probs))
                purchase = .08 + .25 * tastes[u, categories[j]]
                event = rng.choice(['view', 'cart', 'purchase'], p=[.80-purchase, .20, purchase])
                rows.append((f'E{len(rows):07d}', f'U{u:04d}', products.product_id[j], event,
                             START + pd.Timedelta(days=day, seconds=int(rng.integers(0, 86400)))))
    events = pd.DataFrame(rows, columns=['event_id', 'user_id', 'product_id', 'event_type', 'timestamp'])
    events = events.sort_values(['timestamp', 'event_id']).reset_index(drop=True)
    validate(products, events)
    return products, events


def validate(products, events):
    required_p = {'product_id', 'name', 'category', 'style', 'brand', 'price', 'available_at'}
    required_e = {'event_id', 'user_id', 'product_id', 'event_type', 'timestamp'}
    if not required_p.issubset(products) or not required_e.issubset(events):
        raise ValueError('Missing required data columns')
    if products.empty or events.empty:
        raise ValueError('Products and events must be nonempty')
    if products[list(required_p)].isna().any().any() or events[list(required_e)].isna().any().any():
        raise ValueError('Missing values must be handled before training')
    if products.product_id.duplicated().any() or events.event_id.duplicated().any():
        raise ValueError('Duplicate identifiers')
    if not np.isfinite(products.price).all() or (products.price <= 0).any():
        raise ValueError('Prices must be finite and positive')
    if not events.event_type.isin(WEIGHTS).all():
        raise ValueError('Unknown event type')
    if not events.product_id.isin(products.product_id).all():
        raise ValueError('Unknown product reference')
    availability = events.product_id.map(products.set_index('product_id').available_at)
    if (events.timestamp < availability).any():
        raise ValueError('Interaction before product availability')


def save_data(products, events, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    products.to_csv(path / 'products.csv', index=False)
    events.to_csv(path / 'interactions.csv', index=False)


def load_data(directory):
    path = Path(directory)
    products = pd.read_csv(path / 'products.csv', parse_dates=['available_at'])
    events = pd.read_csv(path / 'interactions.csv', parse_dates=['timestamp'])
    validate(products, events)
    return products, events
