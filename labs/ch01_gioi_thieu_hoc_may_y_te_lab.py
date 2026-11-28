# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 1 Code Lab: Pipeline Phân loại Rối loạn Nhịp tim (ECG/HRV)
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi các nguyên tắc: mỗi dòng là một bệnh nhân, Pipeline chống rò rỉ,
# kiểm định chéo trên dữ liệu thô và xử lý mất cân bằng lớp.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

# ---------------------------------------------------------
# 1. KHỞI TẠO TẬP DỮ LIỆU ĐẶC TRƯNG HRV MÔ PHỎNG
# ---------------------------------------------------------
def make_hrv_data(n_normal, n_afib, seed):
    """Mỗi dòng là một bệnh nhân; seed test độc lập với seed train."""
    rng = np.random.default_rng(seed)
    normal = np.column_stack([
        rng.normal(760, 95, n_normal), rng.normal(55, 22, n_normal),
        rng.normal(48, 22, n_normal), rng.normal(28, 12, n_normal)])
    afib = np.column_stack([
        rng.normal(690, 115, n_afib), rng.normal(75, 30, n_afib),
        rng.normal(67, 28, n_afib), rng.normal(38, 16, n_afib)])
    return np.vstack([normal, afib]), np.r_[np.zeros(n_normal, dtype=int),
                                             np.ones(n_afib, dtype=int)]

X_train, y_train = make_hrv_data(560, 140, seed=42)
X_test, y_test = make_hrv_data(240, 60, seed=2026)
feature_names = ['mean_RR', 'SDNN', 'RMSSD', 'pNN50']

print("=" * 60)
print("1. TỔNG QUAN TẬP DỮ LIỆU Y TẾ (HRV FEATURE MATRIX):")
print(f"Tập train: {len(y_train)} dòng (Bình thường = 560, Rung nhĩ = 140)")
print(f"Tập test độc lập: {len(y_test)} dòng (Bình thường = 240, Rung nhĩ = 60)")
print("=" * 60)

# ---------------------------------------------------------
# 2. CÁC MÔ HÌNH SO SÁNH
# ---------------------------------------------------------
def make_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', random_state=42,
                                     max_iter=1000))
    ])

def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall, 'specificity': tn / (tn + fp),
        'precision': precision,
        'f1': 2 * precision * recall / (precision + recall),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

models = {
    'Baseline': None,
    'Proposed': make_pipeline(),
    'Ablation': make_pipeline()
}
predictions = {}

# Baseline: HR = 60000 / mean_RR, không học tham số từ test.
predictions['Baseline'] = (60000 / X_test[:, 0] > 100).astype(int)
models['Proposed'].fit(X_train, y_train)
predictions['Proposed'] = models['Proposed'].predict(X_test)
models['Ablation'].fit(X_train[:, [0]], y_train)
predictions['Ablation'] = models['Ablation'].predict(X_test[:, [0]])

print("2. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (N=300):")
for name, pred in predictions.items():
    result = evaluate(y_test, pred)
    print(f"  - {name}: TN={result['tn']}, FP={result['fp']}, "
          f"FN={result['fn']}, TP={result['tp']}, "
          f"Accuracy={result['accuracy']:.4f}, "
          f"Sensitivity={result['sensitivity']:.4f}, "
          f"Specificity={result['specificity']:.4f}, "
          f"Precision={result['precision']:.4f}, "
          f"F1={result['f1']:.4f}, MCC={result['mcc']:.4f}")

# ---------------------------------------------------------
# 3. KIỂM ĐỊNH CHÉO 5-FOLD (STRATIFIED 5-FOLD CV)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(models['Proposed'], X_train, y_train, cv=cv, scoring='f1')
print(f"3. KIỂM ĐỊNH CHÉO 5-FOLD TRÊN TRAIN (F1-Scores): {cv_scores.round(4)}")
print(f"   Trung bình F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("=" * 60)
