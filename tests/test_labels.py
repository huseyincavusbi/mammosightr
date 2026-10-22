import pandas as pd
from pathlib import Path

LAB = Path(__file__).resolve().parents[1] / "assets/labels"


def test_case_counts():
    c = pd.read_csv(LAB / "cases.csv")
    assert len(c) == 12740
    assert int((c.split != "test").sum()) == 10740
    assert int((c.split == "test").sum()) == 2000


def test_train_birads_distribution():
    c = pd.read_csv(LAB / "cases.csv")
    tr = c[c.split != "test"]
    assert tr.birads3.value_counts().to_dict() == {"1-2": 4432, "4-5": 4125, "0": 2183}


def test_test_birads_distribution():
    c = pd.read_csv(LAB / "cases.csv")
    te = c[c.split == "test"]
    assert te.birads3.value_counts().to_dict() == {"4-5": 896, "1-2": 759, "0": 345}


def test_folds_are_patient_level():
    f = pd.read_csv(LAB / "folds.csv")
    assert len(f) == 10740
    assert f.patient_id.nunique() == 10740
    assert f.fold.value_counts().sort_index().to_dict() == {1: 2685, 2: 2685, 3: 2685, 4: 2685}


def test_image_table():
    i = pd.read_csv(LAB / "images.csv")
    assert len(i) == 42074
    assert set(i.view) == {"CC", "MLO"}
    assert set(i.laterality) == {"left", "right"}
