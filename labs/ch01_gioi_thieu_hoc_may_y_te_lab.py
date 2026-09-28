# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 1 Code Lab: Pipeline Phân loại Rối loạn Nhịp tim (ECG/HRV)
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi các nguyên tắc: Patient-level Split, Proper Scaling, Imbalance Handling.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------------------------------------
# 1. KHỞI TẠO TẬP DỮ LIỆU ĐẶC TRƯNG HRV MÔ PHỎNG (300 BỆNH NHÂN)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 300
n_normal = 240
n_afib = 60

# Nhóm Bình thường: Nhịp tim đều, biến thiên điều hòa
mean_rr_normal = np.random.normal(loc=820, scale=60, size=n_normal)
sdnn_normal = np.random.normal(loc=45, scale=10, size=n_normal)
rmssd_normal = np.random.normal(loc=35, scale=8, size=n_normal)
pnn50_normal = np.random.normal(loc=18, scale=5, size=n_normal)
labels_normal = np.zeros(n_normal, dtype=int)

# Nhóm Rung nhĩ (AFib): Khoảng RR hoàn toàn hỗn loạn, RMSSD và pNN50 tăng cao bất thường
mean_rr_afib = np.random.normal(loc=650, scale=110, size=n_afib)
sdnn_afib = np.random.normal(loc=95, scale=25, size=n_afib)
rmssd_afib = np.random.normal(loc=80, scale=20, size=n_afib)
pnn50_afib = np.random.normal(loc=48, scale=12, size=n_afib)
labels_afib = np.ones(n_afib, dtype=int)

# Hợp nhất DataFrame
df = pd.DataFrame({
    'patient_id': [f"PT_{i:03d}" for i in range(n_samples)],
    'mean_RR': np.concatenate([mean_rr_normal, mean_rr_afib]),
    'SDNN': np.concatenate([sdnn_normal, sdnn_afib]),
    'RMSSD': np.concatenate([rmssd_normal, rmssd_afib]),
    'pNN50': np.concatenate([pnn50_normal, pnn50_afib]),
    'label': np.concatenate([labels_normal, labels_afib])
})

print("=" * 60)
print("1. TỔNG QUAN TẬP DỮ LIỆU Y TẾ (HRV FEATURE MATRIX):")
print(df.head())
print(f"Phân bố lớp: Bình thường = {n_normal} (80%), Rung nhĩ = {n_afib} (20%)")
print("=" * 60)

# ---------------------------------------------------------
# 2. PHÂN CHIA DỮ LIỆU THEO BỆNH NHÂN (PATIENT-LEVEL SPLIT)
# ---------------------------------------------------------
X = df[['mean_RR', 'SDNN', 'RMSSD', 'pNN50']].values
y = df['label'].values

# Phân chia 70% Train - 30% Test, phân tầng theo nhãn y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 3. TIỀN XỬ LÝ: CHUẨN HÓA Z-SCORE (FIT CHỈ TRÊN TRAIN)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Áp dụng mu, sigma của Train sang Test

# ---------------------------------------------------------
# 4. HUẤN LUYỆN MÔ HÌNH PROPOSED (LOGISTIC REGRESSION CÂN BẰNG)
# ---------------------------------------------------------
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 5. ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)

# Tính toán ma trận nhầm lẫn
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print("2. KẾT QUẢ ĐÁNH GIÁ LÂM SÀNG TRÊN TẬP TEST (N=90):")
print(f"  - True Positives (TP):  {tp}")
print(f"  - False Negatives (FN): {fn}  (Ca bệnh bị bỏ sót)")
print(f"  - False Positives (FP): {fp}  (Báo động giả)")
print(f"  - True Negatives (TN):  {tn}")
print("-" * 60)
print(f"  - Độ chính xác (Accuracy):  {acc:.4f}")
print(f"  - Độ nhạy (Sensitivity):    {sensitivity:.4f}")
print(f"  - Độ đặc hiệu (Specificity): {specificity:.4f}")
print(f"  - Độ chuẩn xác (Precision):  {precision:.4f}")
print(f"  - Điểm F1 (F1-Score):        {f1:.4f}")
print("-" * 60)
print("BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
print(classification_report(y_test, y_pred, target_names=['Bình thường', 'Rung nhĩ']))

# ---------------------------------------------------------
# 6. KIỂM ĐỊNH CHÉO 5-FOLD (STRATIFIED 5-FOLD CV)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=cv, scoring='f1')
print(f"3. KIỂM ĐỊNH CHÉO 5-FOLD (F1-Scores): {cv_scores.round(4)}")
print(f"   Trung bình F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("=" * 60)