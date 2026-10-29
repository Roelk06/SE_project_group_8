"""Shared helpers for the test suite (toy data, loading functions, running scripts)."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, save_npz

# project folder = the folder that contains tests/, models/, data/ ...
ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CLASS_NAMES = ["ham", "spam"]


# ============================================================
# Finding / loading the model scripts
# ============================================================

def find_script(script_name):
    """Looks for a model script in models/ first, then in the project folder."""
    for folder in (ROOT / "models", ROOT):
        path = folder / script_name
        if path.exists():
            return path
    pytest.fail(f"Could not find {script_name} in {ROOT / 'models'} or {ROOT}")


def load_prediction_function(script_name):
    """
    Gets predict_with_confidence (and CLASS_NAMES) out of a model script
    WITHOUT running the script.

    The model scripts train a model as soon as they are run/imported, so a normal
    "import sms_model" would start training. Instead we read the file, pick out just
    the function and CLASS_NAMES, and run only those.
    """
    path = find_script(script_name)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    wanted = []
    found_function = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "predict_with_confidence":
            wanted.append(node)
            found_function = True
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CLASS_NAMES"
            for target in node.targets
        ):
            wanted.append(node)

    if not found_function:
        pytest.fail(f"{script_name} has no predict_with_confidence function")

    module = ast.Module(body=wanted, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["predict_with_confidence"], namespace.get("CLASS_NAMES")


# ============================================================
# Checking the standardized output
# ============================================================

def assert_standard_result(result):
    """One prediction must look like {"class": "spam", "confidence": 0.93}."""
    assert isinstance(result, dict), f"result is not a dict: {result!r}"
    assert set(result.keys()) == {"class", "confidence"}, f"wrong keys: {list(result)}"
    assert result["class"] in EXPECTED_CLASS_NAMES, f"bad class: {result['class']!r}"
    assert isinstance(result["confidence"], float), "confidence must be a float"
    assert 0.0 <= result["confidence"] <= 1.0, f"confidence out of range: {result['confidence']}"


# ============================================================
# Toy data (same file formats the real processed data uses)
# ============================================================

def make_sparse_data(n_rows, n_features=30, seed=0):
    """Small sparse matrix + 0/1 labels. Spam rows have extra weight on the first 5 features."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n_rows, dtype=np.int64)
    y[: n_rows // 3] = 1
    rng.shuffle(y)

    dense = (rng.random((n_rows, n_features)) < 0.1) * rng.uniform(0.2, 1.0, (n_rows, n_features))
    n_spam = int((y == 1).sum())
    dense[y == 1, :5] += (rng.random((n_spam, 5)) < 0.8) * rng.uniform(0.5, 1.0, (n_spam, 5))
    return csr_matrix(dense), y


def write_sparse_dataset(folder, train=150, val=50, test=50):
    """Writes x_*.npz / y_*.npy like the SMS, email and URL pipelines do."""
    folder.mkdir(parents=True, exist_ok=True)
    for seed, (split, n) in enumerate([("train", train), ("val", val), ("test", test)]):
        X, y = make_sparse_data(n, seed=seed)
        save_npz(folder / f"x_{split}.npz", X)
        np.save(folder / f"y_{split}.npy", y)


def write_image_dataset(folder, train=64, val=32, test=32, size=32):
    """Writes x_*.npy shaped (samples, 3, size, size) / y_*.npy like the image pipeline does."""
    folder.mkdir(parents=True, exist_ok=True)
    for seed, (split, n) in enumerate([("train", train), ("val", val), ("test", test)]):
        rng = np.random.default_rng(seed)
        y = np.zeros(n, dtype=np.int64)
        y[: n // 3] = 1
        rng.shuffle(y)
        x = rng.normal(0, 1, size=(n, 3, size, size)).astype(np.float32)
        for i in np.where(y == 1)[0]:
            x[i, :, 4:20, 4:20] += 2.0  # bright square = "spam"
        np.save(folder / f"x_{split}.npy", x)
        np.save(folder / f"y_{split}.npy", y)


# ============================================================
# Running a model script in a throw-away project folder
# ============================================================

def run_script_in_sandbox(script_name, data_folder, data_writer, tmp_path,
                          text_edits=None, timeout=300):
    """
    Builds   <tmp>/project/models/<script>   and   <tmp>/project/data/processed/<data_folder>/
    (the same layout as the real project), then runs the script from inside models/.
    Real data and real saved models are never touched.

    text_edits: optional list of (old, new) replacements applied to the COPY of the script
    (used by the image test to skip downloading pretrained weights).
    """
    project = tmp_path / "project"
    models_dir = project / "models"
    models_dir.mkdir(parents=True)

    source = find_script(script_name).read_text(encoding="utf-8")
    for old, new in (text_edits or []):
        assert old in source, f"text not found in {script_name}: {old!r}"
        source = source.replace(old, new)
    (models_dir / script_name).write_text(source, encoding="utf-8")

    data_writer(project / "data" / "processed" / data_folder)

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=models_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    return result, project
