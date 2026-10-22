"""Build canonical label tables and patient-level folds.

Outputs (labels/):
  cases.csv   - 12,740 rows: patient_id, split, birads5, birads3, density, quadrant_right, quadrant_left, age_bin
  images.csv  - 42,074 train rows: patient_id, shard, row_in_shard, image_filename, view, laterality, birads5_img, birads5, birads3, density, quadrant
  folds.csv   - 10,740 train cases with fold 1..4 (patient-level, stratified by birads3+density)
  qa.md       - sanity report
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from openpyxl import load_workbook
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PARQUET = sorted(glob.glob(str(ROOT / "preprocessed/data/*.parquet")))
XLSX = ROOT / "labels/supplementary"
OUT = ROOT / "labels"

PARQ_COLS = ["patient_id", "image_filename", "view", "laterality", "birads",
             "breast_composition", "findings_locations", "patient_birads"]
QUAD_MAP = {"Center": "CENTER", "Upper Outer": "UOQ", "Upper Inner": "UIQ",
            "Lower Outer": "LOQ", "Lower Inner": "LIQ"}
QUAD_KEYS = {"UOQ", "UIQ", "LOQ", "LIQ", "CENTER"}
BIRADS3 = {0: "0", 1: "1-2", 2: "1-2", 4: "4-5", 5: "4-5"}
CAT3 = {"BI-RADS0": "0", "BI-RADS1-2": "1-2", "BI-RADS4-5": "4-5"}


def norm_case(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) and not np.isnan(v):
        return str(int(v))
    return str(v).strip()


def parse_quad(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        tokens = [str(t).strip() for t in json.loads(s)]
    except Exception:
        tokens = re.findall(r"UOQ|UIQ|LOQ|LIQ|CENTER", s.upper())
    return sorted({t.upper() for t in tokens if t.upper() in QUAD_KEYS})


def read_xlsx(path):
    wb = load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) if h is not None else f"col{i}" for i, h in enumerate(rows[0])]
    return pd.DataFrame([dict(zip(hdr, r)) for r in rows[1:]], columns=hdr)


def col(df, prefix):
    for c in df.columns:
        if c.startswith(prefix):
            return c
    raise KeyError(prefix)


def main():
    OUT.mkdir(exist_ok=True)
    case_frames = []
    for name, split in [("Supplementary_TRAIN1.xlsx", "train1"),
                        ("Supplementary_TRAIN2.xlsx", "train2"),
                        ("Supplementary_TESTSET.xlsx", "test")]:
        x = read_xlsx(XLSX / name)
        cat = x[col(x, "BIRADS CATEGORY")].astype(str).str.strip()
        birads5 = None
        if any(c.startswith("BI-RADS 0/1/2/4/5") for c in x.columns):
            birads5 = (x[col(x, "BI-RADS 0/1/2/4/5")].astype(str)
                       .str.replace("BI-RADS ", "", regex=False).astype(int))
        out = pd.DataFrame({
            "patient_id": x[col(x, "CASENUMBER")].map(norm_case),
            "split": split,
            "birads3": cat.map(CAT3),
            "density": x[col(x, "BREAST DENSITY")].astype(str).str.strip(),
            "quadrant_right": x[col(x, "QUADRANT INFORMATION (RIGHT)")].map(lambda v: "|".join(parse_quad(v))),
            "quadrant_left": x[col(x, "QUADRANT INFORMATION (LEFT)")].map(lambda v: "|".join(parse_quad(v))),
        })
        if birads5 is not None:
            out.insert(2, "birads5", birads5)
        case_frames.append(out)
    cases = pd.concat(case_frames, ignore_index=True)
    cases["birads5"] = cases["birads5"].astype("Int64")

    age = read_xlsx(XLSX / "age_bins_with_decades_for suppl..xlsx")
    age["patient_id"] = age["CaseNumber"].map(norm_case)
    cases = cases.merge(age[["patient_id", "Age_Bin"]].rename(columns={"Age_Bin": "age_bin"}),
                        on="patient_id", how="left")

    img_frames = []
    for f in PARQUET:
        t = pq.read_table(f, columns=PARQ_COLS).to_pandas()
        t["shard"] = os.path.basename(f)
        t["row_in_shard"] = np.arange(len(t))
        img_frames.append(t)
    img = pd.concat(img_frames, ignore_index=True)
    img["quadrant"] = img["findings_locations"].map(
        lambda lst: "|".join(sorted({QUAD_MAP.get(str(x), str(x)) for x in lst})))
    img = img.rename(columns={"birads": "birads5_img", "breast_composition": "density",
                              "patient_birads": "birads5"})
    img["birads3"] = img["birads5"].map(BIRADS3)

    img_out = img[["patient_id", "shard", "row_in_shard", "image_filename", "view",
                   "laterality", "birads5_img", "birads5", "birads3", "density", "quadrant"]]
    cases_out = cases[["patient_id", "split", "birads5", "birads3", "density",
                       "quadrant_right", "quadrant_left", "age_bin"]]

    train = cases_out[cases_out.split != "test"].copy()
    strat = train.birads3 + "_" + train.density
    fold = np.zeros(len(train), dtype=int)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for k, (_, vi) in enumerate(skf.split(train, strat), start=1):
        fold[vi] = k
    train["fold"] = fold
    folds = train[["patient_id", "fold", "birads5", "birads3", "density"]]

    cases_out.to_csv(OUT / "cases.csv", index=False)
    img_out.to_csv(OUT / "images.csv", index=False)
    folds.to_csv(OUT / "folds.csv", index=False)

    pq_side = (img.groupby(["patient_id", "laterality"])["quadrant"]
               .apply(lambda s: "|".join(sorted({q for v in s for q in v.split("|") if q}))))
    px = cases.set_index("patient_id")
    bad = []
    for (pid, side), tags in pq_side.items():
        if pid not in px.index or px.loc[pid, "split"] == "test":
            continue
        want = px.loc[pid, {"right": "quadrant_right", "left": "quadrant_left"}[side]]
        want = "" if pd.isna(want) else want
        if tags != want:
            bad.append((pid, side, tags, want))

    lines = ["# Label QA report", "", "## Counts"]
    lines.append(f"- cases: {len(cases_out)} (train {len(train)}, test {(cases_out.split=='test').sum()})")
    lines.append(f"- images: {len(img_out)} | patients: {img_out.patient_id.nunique()}")
    lines.append(f"- images per patient: {dict(img.groupby('patient_id').size().value_counts().sort_index())}")
    lines.append("")
    lines.append("## Case-level distributions (train / test)")
    for c in ["birads3", "density"]:
        tr = cases_out[cases_out.split != "test"][c].value_counts().sort_index().to_dict()
        te = cases_out[cases_out.split == "test"][c].value_counts().sort_index().to_dict()
        lines.append(f"- {c}: train {tr} | test {te}")
    lines.append(f"- age_bin missing: {cases_out.age_bin.isna().sum()}")
    lines.append("")
    lines.append("## Folds (patient-level, stratified by birads3+density)")
    lines.append(f"- sizes: {dict(folds.fold.value_counts().sort_index())}")
    xt = pd.crosstab(folds.fold, folds.birads3)
    lines.append("- fold x birads3:")
    lines.extend(["```", xt.to_string(), "```"])
    lines.append("")
    lines.append("## Quadrant")
    dev = cases_out[cases_out.split != "test"]
    tst = cases_out[cases_out.split == "test"]
    for name, sub in [("train", dev), ("test", tst)]:
        for side in ["quadrant_right", "quadrant_left"]:
            lines.append(f"- {name} {side} non-empty: {sub[side].ne('').sum()}")
        both = ((sub.quadrant_right.ne("")) & (sub.quadrant_left.ne(""))).sum()
        lines.append(f"- {name} both sides non-empty: {both}")
    multi = cases_out[(cases_out.quadrant_right.str.contains(r"\|")) | (cases_out.quadrant_left.str.contains(r"\|"))]
    lines.append(f"- cases with multi-quadrant side: {len(multi)}")
    lines.append(f"- official-vs-parquet quadrant mismatches (train): {len(bad)}")
    for b in bad[:25]:
        lines.append(f"  - {b}")
    lines.append("")
    lines.append("## Files")
    lines.append("- cases.csv / images.csv / folds.csv written")
    (OUT / "qa.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
