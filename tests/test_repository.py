from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def test_required_project_files_exist():
    required = [
        "README.md",
        "requirements.txt",
        "app/api.py",
        "app/dlp_core.py",
        "src/train_xgb.py",
        "src/logistic_regression.py",
        "artifacts/policy.json",
        "artifacts_xgb/xgboost_classifier.joblib",
        "artifacts_xgb/tfidf_vectorizer_xgb.joblib",
        "artifacts_xgb/label_encoder.joblib",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"Missing required files: {missing}"

def test_policy_json_is_valid():
    path = ROOT / "artifacts" / "policy.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    assert isinstance(data, (dict, list))

def test_no_generated_bert_cache_is_committed():
    assert not (ROOT / "artifacts" / "bert_embeddings.npy").exists()

def test_no_editor_temp_file():
    assert not (ROOT / "src" / "tempCodeRunnerFile.py").exists()
