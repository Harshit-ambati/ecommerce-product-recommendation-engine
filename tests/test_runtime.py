from pathlib import Path
from recommender import runtime


def test_clean_deployment_generates_model(monkeypatch, tmp_path):
    artifact = tmp_path / 'artifacts' / 'recommender.pkl'
    calls = []

    def generate(root):
        calls.append(Path(root))
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'generated')

    monkeypatch.setattr(runtime, 'run', generate)
    monkeypatch.setattr(runtime.Recommender, 'load', lambda path: Path(path).read_bytes())
    assert runtime.ensure_model(tmp_path) == b'generated'
    assert runtime.ensure_model(tmp_path) == b'generated'
    assert calls == [tmp_path]
