# -*- coding: utf-8 -*-
# Chương 1 §1.9.1 — Tái lập rò rỉ nhãn qua biến `time` trên UCI Heart Failure Clinical Records
# (Chicco & Jurman, 2020; N = 299). Random Forest 300 cây, Stratified 5-fold lặp 5 lần, random_state=0.
# Kết quả tham chiếu (chạy 14/11/2026, scikit-learn 1.9.1): ROC-AUC 0.7999 (bỏ `time`) vs 0.9080 (giữ `time`).
#
# Cách chạy:  python ch01_uci_hf_time_leakage.py [đường_dẫn_csv]
# Mặc định đọc CSV trong repo HMYT; nếu không có, tải bản công khai trên portal.

import sys
import pathlib
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

LOCAL = pathlib.Path('labs_chuyen_de/ch02_suy_tim_risk_dxai/data/heart_failure_clinical_records_dataset.csv')
PUBLIC_URL = ('https://fossbk-spec.github.io/hmyt-book/labs_chuyen_de/ch02_suy_tim_risk_dxai/'
              'data/heart_failure_clinical_records_dataset.csv')

def load_data(path_arg=None):
    if path_arg:
        return pd.read_csv(path_arg)
    if LOCAL.exists():
        return pd.read_csv(LOCAL)
    cache = pathlib.Path('heart_failure_clinical_records_dataset.csv')
    if not cache.exists():
        print(f"[i] Tải dữ liệu từ {PUBLIC_URL}")
        urllib.request.urlretrieve(PUBLIC_URL, cache)
    return pd.read_csv(cache)

df = load_data(sys.argv[1] if len(sys.argv) > 1 else None)
df.columns = [c.lower() for c in df.columns]        # bản UCI gốc dùng DEATH_EVENT, bản repo dùng death_event
target = 'death_event'
y = df[target]
features_all = [c for c in df.columns if c != target]
features_clean = [c for c in features_all if c != 'time']

print("=" * 70)
print(f"UCI Heart Failure Clinical Records: N = {len(df)}, tử vong = {int(y.sum())} ({y.mean()*100:.1f}%)")
print("Mô hình: RandomForestClassifier(n_estimators=300, random_state=0)")
print("Kiểm định: RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0), scoring='roc_auc'")
print("=" * 70)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
results = {}
for label, cols in (("Mô hình sạch (bỏ `time`)", features_clean), ("Mô hình rò rỉ (giữ `time`)", features_all)):
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    scores = cross_val_score(rf, df[cols], y, cv=cv, scoring='roc_auc')
    results[label] = scores.mean()
    print(f"  - {label:<30}: ROC-AUC trung bình = {scores.mean():.4f} (+/- {scores.std():.4f}), {len(cols)} đặc trưng")

diff = results["Mô hình rò rỉ (giữ `time`)"] - results["Mô hình sạch (bỏ `time`)"]
print("-" * 70)
print(f"Chênh lệch do rò rỉ biến `time`: {diff:+.4f} ROC-AUC")
print("Biến `time` (số ngày theo dõi) chỉ biết sau khi đã quan sát kết cục, nên không được dùng làm đặc trưng dự đoán.")
print("=" * 70)
