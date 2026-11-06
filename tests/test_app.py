from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(not Path('artifacts/recommender.pkl').exists(), reason='Run training pipeline before UI integration tests')
def test_demo_loads_and_changes_shopper():
    app = AppTest.from_file('app.py', default_timeout=60).run()
    assert not app.exception
    assert app.selectbox[0].value == 'New shopper'
    app.selectbox[0].select('U0000').run()
    app.selectbox[1].select('collaborative').run()
    assert not app.exception
    assert app.metric[0].value != '0'
