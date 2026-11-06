"""Three ranked recommenders with immutable prediction-time history."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data import WEIGHTS, validate

METHODS = ('popularity', 'content', 'collaborative')


class Recommender:
    def fit(self, products, events, cutoff):
        validate(products, events)
        self.cutoff = pd.Timestamp(cutoff)
        self.products = products[products.available_at <= self.cutoff].sort_values('product_id').reset_index(drop=True).copy()
        self.events = events[(events.timestamp < self.cutoff) & events.product_id.isin(self.products.product_id)].copy()
        if self.products.empty or self.events.empty:
            raise ValueError('Training requires available products and prior interactions')
        self.item_index = {p: i for i, p in enumerate(self.products.product_id)}
        self.users = sorted(self.events.user_id.unique())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.histories = {u: g.groupby('product_id').event_type.apply(lambda s: sum(WEIGHTS[x] for x in s)).to_dict()
                          for u, g in self.events.groupby('user_id')}
        user_rows, item_cols, values = [], [], []
        for u, history in self.histories.items():
            for p, weight in history.items():
                user_rows.append(self.user_index[u])
                item_cols.append(self.item_index[p])
                values.append(np.log1p(weight))
        self.matrix = csr_matrix((values, (user_rows, item_cols)), shape=(len(self.users), len(self.products)))
        # Distinct engaged shoppers, not future events or global dataset counts.
        self.popularity = np.asarray((self.matrix > 0).sum(axis=0)).ravel().astype(float)
        text = self.products[['category', 'style', 'brand']].agg(' '.join, axis=1)
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.features = self.vectorizer.fit_transform(text)
        self.content_similarity = cosine_similarity(self.features)
        self.collaborative_similarity = cosine_similarity(self.matrix.T)
        np.fill_diagonal(self.collaborative_similarity, 0)
        self.selected_method = 'content'
        return self

    def recommend(self, user_id, k=10, method=None):
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50:
            raise ValueError('k must be an integer between 1 and 50')
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError('user_id must be a nonempty string')
        method = method or self.selected_method
        if method not in METHODS:
            raise ValueError(f'Unknown recommendation method: {method}')
        history = self.histories.get(user_id, {})
        indices = np.array([self.item_index[p] for p in history], dtype=int)
        weights = np.log1p(list(history.values()))
        similarity = None
        if method == 'popularity' or not history:
            scores = self.popularity.copy()
        else:
            similarity = self.content_similarity if method == 'content' else self.collaborative_similarity
            scores = (weights @ similarity[indices]) / weights.sum()
        candidates = [i for i in range(len(self.products)) if i not in set(indices)]
        # Popularity is a deterministic tie-break and zero-signal fallback.
        ranked = sorted(candidates, key=lambda i: (-scores[i], -self.popularity[i], self.products.product_id[i]))[:k]
        output = []
        for rank, i in enumerate(ranked, 1):
            p = self.products.iloc[i]
            reason = 'Popular among shoppers in the training history'
            fallback = method != 'popularity' and (not history or scores[i] <= 0)
            source_id = None
            if similarity is not None and scores[i] > 0:
                contribution = weights * similarity[indices, i]
                source = self.products.iloc[indices[int(np.argmax(contribution))]]
                source_id = source.product_id
                if method == 'content':
                    shared = [str(p[c]) for c in ['category', 'style', 'brand'] if p[c] == source[c]]
                    reason = f'Shares {", ".join(shared)} with {source["name"]}'
                else:
                    reason = f'Shoppers who engaged with {source["name"]} also engaged with this product'
            output.append({'rank': rank, 'product_id': p.product_id, 'name': p['name'],
                           'category': p.category, 'price': float(p.price), 'score': round(float(scores[i]), 6),
                           'reason': reason, 'source_product_id': source_id, 'fallback': bool(fallback)})
        return output

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        # Only load artifacts generated by this project's training command.
        with Path(path).open('rb') as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise ValueError('Unexpected artifact type')
        return model
