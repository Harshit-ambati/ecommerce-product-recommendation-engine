"""Interactive review of real saved-artifact recommendations."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from recommender.model import METHODS
from recommender.runtime import ensure_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Product Discovery Lab', page_icon='🛍️', layout='wide')
st.title('Product Discovery Lab')
st.caption('E-commerce recommendation engine · AI & Machine Learning assessment')
st.info('Synthetic shoppers and products. This demo measures an offline simulation; scores are ranking signals, not purchase probabilities.')
@st.cache_resource(show_spinner='Preparing the recommendation model and evaluation data…')
def load_model(root):
    return ensure_model(root)


model = load_model(str(ROOT))
with st.sidebar:
    st.header('Explore recommendations')
    shopper = st.selectbox('Shopper', ['New shopper'] + model.users)
    user = 'new-shopper' if shopper == 'New shopper' else shopper
    method = st.selectbox('Recommendation method', list(METHODS), index=list(METHODS).index(model.selected_method))
    k = st.slider('Number of products', 1, 20, 10)
    st.caption(f'Chosen by validation: {model.selected_method}')
    st.caption(f'History cutoff: {model.cutoff.date()}. Catalog and shopper history are frozen at this point.')

discover, evidence, data_tab = st.tabs(['Discover', 'Model evidence', 'Data overview'])
with discover:
    history = model.events[model.events.user_id == user]
    cols = st.columns(3)
    cols[0].metric('Past interactions', len(history))
    cols[1].metric('Products explored', history.product_id.nunique())
    cols[2].metric('Available catalog', len(model.products))
    if history.empty:
        st.info('This shopper has no prior history. Recommendations use the popularity fallback.')
    results = model.recommend(user, k, method)
    st.subheader('Recommended products')
    st.caption('Previously explored products are excluded. Reasons identify a contributing signal, not a causal explanation.')
    for start in range(0, len(results), 3):
        for column, product in zip(st.columns(3), results[start:start+3]):
            with column.container(border=True):
                st.caption(f'#{product["rank"]} · {product["category"]} · {product["product_id"]}')
                st.subheader(product['name'])
                st.write(f'Price: {product["price"]:.2f} synthetic currency units')
                st.write(product['reason'])
                st.caption(f'Ranking score: {product["score"]:.4f}' + (' · Popularity fallback' if product['fallback'] else ''))
    st.download_button('Download recommendations', json.dumps(results, indent=2), 'recommendations.json', 'application/json')
    with st.expander('Shopper history and contributing products'):
        st.dataframe(history.merge(model.products[['product_id', 'name', 'category']], on='product_id').sort_values('timestamp', ascending=False), hide_index=True)

with evidence:
    st.subheader('Held-out ranking quality')
    st.write('Models are selected using validation NDCG@10, then refitted on training + validation history. The final 30 days are held out for reporting.')
    metrics_path = ROOT / 'reports' / 'test_metrics.csv'
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        overall = metrics[metrics.segment == 'all'].set_index('method')
        st.bar_chart(overall[['precision@10', 'recall@10', 'ndcg@10']])
        st.dataframe(metrics, hide_index=True)
        st.caption('Precision: relevant items per 10 recommendations. Recall: fraction of eligible targets found. NDCG: relevance weighted by rank. Coverage: fraction of the catalog recommended across evaluated shoppers.')
        st.write('Cold users receive popularity recommendations. Content features can score new products; collaborative filtering cannot personalize products with no interaction history.')
        with st.expander('Validation results and evaluation exclusions'):
            st.dataframe(pd.read_csv(ROOT / 'reports' / 'validation_metrics.csv'), hide_index=True)
            st.json(json.loads((ROOT / 'reports' / 'run_manifest.json').read_text()))
        with st.expander('Successful and unsuccessful recommendation examples'):
            st.json(json.loads((ROOT / 'reports' / 'error_cases.json').read_text()))
    st.warning('Offline clicks and carts are incomplete preference labels. The generator encodes category/style preferences, which favors methods using those signals. Real-world usefulness requires licensed real data and prospective validation.')

with data_tab:
    st.subheader('Reproducible synthetic shopping activity')
    eda_path = ROOT / 'reports' / 'eda.json'
    if eda_path.exists():
        eda = json.loads(eda_path.read_text())
        cols = st.columns(3)
        cols[0].metric('Total events', f'{eda["events"]:,}')
        cols[1].metric('Simulated shoppers', eda['users'])
        cols[2].metric('Catalog products', eda['products'])
        st.line_chart(pd.read_csv(ROOT / 'reports' / 'daily_events.csv', index_col=0))
        st.bar_chart(pd.read_csv(ROOT / 'reports' / 'category_events.csv', index_col=0))
        with st.expander('Quality checks and split sizes'):
            st.json(eda)
    st.write('No real personal data is used. Repeated user-product interactions are valid separate events; duplicate event IDs are rejected. Product prices are synthetic.')
