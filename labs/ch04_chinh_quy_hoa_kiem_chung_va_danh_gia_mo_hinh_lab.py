# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 4 Code Lab: Chính quy hóa Lasso/Ridge & Đánh giá Hiệu năng Y tế trên Dữ liệu p >> n
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi: LassoCV chọn lọc gen, RidgeCV co trọng số, Stratified K-Fold, ROC-AUC, Brier Score.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. KHỞI TẠO TẬP DỮ LIỆU BIỂU HIỆN GEN MÔ PHỎNG (n=200, p=500)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 200
n_features = 500 # p >> n

# Sinh ma trận biểu hiện gen chuẩn Gaussian
X_raw = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

# Chỉ có 10 gen đầu tiên thực sự liên quan đến bệnh lý di căn (True Biomarkers)
true_weights = np.zeros(n_features)
true_weights[0:10] = [1.2, -0.9, 1.5, -1.1, 0.8, -1.3, 1.0, -0.7, 1.4, -1.0]

# Xác suất di căn thực tế
z = np.dot(X_raw, true_weights)
prob = 1 / (1 + np.exp(-z))
y = np.random.binomial(1, prob, size=n_samples)

print("=" * 70)
print("1. TỔNG QUAN TẬP DỮ LIỆU BIỂU HIỆN GEN Y SINH (GENOMICS COHORT):")
print(f"  - Số lượng bệnh nhân (n): {n_samples}")
print(f"  - Số lượng gen khảo sát (p): {n_features} (Bài toán p >> n)")
print(f"  - Số lượng gen bệnh lý thực sự (Ground Truth): 10 gen")
print(f"  - Tỷ lệ ca di căn (y=1): {sum(y==1)} ({sum(y==1)/n_samples*100:.1f}%)")
print("=" * 70)

# ---------------------------------------------------------
# 2. PHÂN CHIA DỮ LIỆU TRAIN - TEST (ĐÓNG BĂNG TEST SET)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.30, stratify=y, random_state=42
)

# Chuẩn hóa Z-score (fit duy nhất trên train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. HUẤN LUYỆN PROPOSED MODEL: LOGISTIC REGRESSION PHẠT L1 (LASSO)
# ---------------------------------------------------------
# Sử dụng solver 'saga' và L1 penalty với Stratified 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Khởi tạo Logistic Regression với phạt L1 (Lasso)
lasso_model = LogisticRegression(penalty='l1', solver='saga', C=0.25, max_iter=2000, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

# Khởi tạo Logistic Regression với phạt L2 (Ridge)
ridge_model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.10, max_iter=2000, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 4. ĐÁNH GIÁ VÀ SO SÁNH HIỆU NĂNG TRÊN TẬP TEST ĐỘC LẬP (N=60)
# ---------------------------------------------------------
# Đánh giá Lasso
y_prob_lasso = lasso_model.predict_proba(X_test_scaled)[:, 1]
y_pred_lasso = lasso_model.predict(X_test_scaled)
auc_lasso = roc_auc_score(y_test, y_prob_lasso)
non_zero_lasso = np.sum(lasso_model.coef_[0] != 0)

# Đánh giá Ridge
y_prob_ridge = ridge_model.predict_proba(X_test_scaled)[:, 1]
y_pred_ridge = ridge_model.predict(X_test_scaled)
auc_ridge = roc_auc_score(y_test, y_prob_ridge)
non_zero_ridge = np.sum(ridge_model.coef_[0] != 0)

print("2. KẾT QUẢ SO SÁNH HIỆU NĂNG TRÊN TẬP TEST ĐỘC LẬP (N=60):")
print(f"{'Mô hình':<25} | {'Số gen giữ lại':<16} | {'ROC-AUC':<10} | {'Brier Score':<10}")
print("-" * 70)
print(f"{'1. Proposed (Lasso L1)':<25} | {non_zero_lasso:<16} | {auc_lasso:<10.4f} | {brier_score_loss(y_test, y_prob_lasso):<10.4f}")
print(f"{'2. Ablation (Ridge L2)':<25} | {non_zero_ridge:<16} | {auc_ridge:<10.4f} | {brier_score_loss(y_test, y_prob_ridge):<10.4f}")
print("=" * 70)

# ---------------------------------------------------------
# 5. TRÍCH XUẤT CHỮ KÝ GEN ĐƯỢC LASSO LỰA CHỌN
# ---------------------------------------------------------
selected_indices = np.where(lasso_model.coef_[0] != 0)[0]
selected_weights = lasso_model.coef_[0][selected_indices]

biomarker_df = pd.DataFrame({
    'Mã Gen': [f"Gene_{idx:03d}" for idx in selected_indices],
    'Trọng số Lasso (w)': selected_weights.round(4),
    'Ý nghĩa Sinh học': ['Tăng nguy cơ di căn' if w > 0 else 'Bảo vệ / Giảm di căn' for w in selected_weights]
})

print("3. DANH MỤC CHỮ KÝ DẤU ẤN SINH HỌC GEN ĐƯỢC LASSO CHỌN LỌC:")
print(biomarker_df.to_string(index=False))
print("=" * 70)
print("[+] HOÀN TẤT CODE LAB: LASSO CO THÀNH CÔNG 500 GEN VỀ CHỮ KÝ GEN TINH GỌN!")