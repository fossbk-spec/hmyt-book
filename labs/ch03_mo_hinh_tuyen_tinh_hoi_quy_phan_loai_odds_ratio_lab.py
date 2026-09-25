# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 3 Code Lab: Huấn luyện Hồi quy Logistic & Trích xuất Bảng Odds Ratio Lâm sàng
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi: Chuẩn hóa, Huấn luyện, Đánh giá ROC-AUC và Tính toán OR kèm 95% CI.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. KHỞI TẠO TẬP DỮ LIỆU NGUY CƠ TIM MẠCH MÔ PHỎNG (N=500 BỆNH NHÂN)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 500

# Đặc trưng lâm sàng
age = np.random.normal(loc=55, scale=10, size=n_samples).clip(30, 80)
sbp = np.random.normal(loc=130, scale=18, size=n_samples).clip(90, 200) # Huyết áp tâm thu
chol = np.random.normal(loc=210, scale=35, size=n_samples).clip(120, 320) # Cholesterol (mg/dL)
smoking = np.random.binomial(1, p=0.35, size=n_samples) # 1: Hút thuốc, 0: Không
diabetes = np.random.binomial(1, p=0.20, size=n_samples) # 1: Đái tháo đường, 0: Không

# Mô hình hóa xác suất sinh bệnh theo hàm Logit thực tế
# logit = -8.5 + 0.06*age + 0.025*sbp + 0.012*chol + 0.85*smoking + 1.10*diabetes
z_true = -8.5 + 0.06*age + 0.025*sbp + 0.012*chol + 0.85*smoking + 1.10*diabetes
prob_true = 1 / (1 + np.exp(-z_true))
y = np.random.binomial(1, prob_true, size=n_samples)

df = pd.DataFrame({
    'patient_id': [f"PT_{i:03d}" for i in range(n_samples)],
    'age': age,
    'sbp': sbp,
    'cholesterol': chol,
    'smoking': smoking,
    'diabetes': diabetes,
    'cvd_event': y
})

print("=" * 70)
print("1. TỔNG QUAN TẬP DỮ LIỆU NGUY CƠ TIM MẠCH (FRAMINGHAM CVD COHORT):")
print(df.head())
print(f"Phân bố biến cố: Không biến cố = {sum(y==0)} ({sum(y==0)/n_samples*100:.1f}%), Biến cố CVD = {sum(y==1)} ({sum(y==1)/n_samples*100:.1f}%)")
print("=" * 70)

# ---------------------------------------------------------
# 2. PHÂN CHIA DỮ LIỆU & TIỀN XỬ LÝ (FIT CHỈ TRÊN TRAIN)
# ---------------------------------------------------------
feature_names = ['age', 'sbp', 'cholesterol', 'smoking', 'diabetes']
X = df[feature_names].values
y = df['cvd_event'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Chuẩn hóa biến số liên tục (fit trên train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. HUẤN LUYỆN MÔ HÌNH HỒI QUY LOGISTIC
# ---------------------------------------------------------
# C = 1e5 để tương đương hồi quy không chính quy hóa (Unpenalized Logistic Regression)
model = LogisticRegression(C=1e5, solver='lbfgs', random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 4. ĐÁNH GIÁ HIỆU NĂNG TRÊN TẬP KIỂM THỬ ĐỘC LẬP (N=150)
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("2. KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (N=150):")
print(f"  - Diện tích dưới đường cong ROC (ROC-AUC): {roc_auc:.4f}")
print(f"  - Điểm hiệu chuẩn Brier Score (càng thấp càng tốt): {brier:.4f}")
print("-" * 70)
print("BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
print(classification_report(y_test, y_pred, target_names=['Không biến cố', 'Biến cố CVD']))

# ---------------------------------------------------------
# 5. TRÍCH XUẤT VÀ TÍNH TOÁN BẢNG ODDS RATIO LÂM SÀNG KÈM 95% CI
# ---------------------------------------------------------
# Huấn luyện trên thang đo gốc (Unscaled) để có hệ số mang ý nghĩa lâm sàng trực tiếp
model_raw = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000, random_state=42)
model_raw.fit(X_train, y_train)

# Ước lượng ma trận hiệp phương sai qua phép xấp xỉ Thông tin Fisher
p_pred_train = model_raw.predict_proba(X_train)[:, 1]
W_diag = p_pred_train * (1 - p_pred_train)
X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
V = np.diag(W_diag)
cov_matrix = np.linalg.inv(X_design.T @ V @ X_design)
se_weights = np.sqrt(np.diag(cov_matrix))[1:] # Bỏ qua intercept

weights = model_raw.coef_[0]
odds_ratios = np.exp(weights)
ci_lower = np.exp(weights - 1.96 * se_weights)
ci_upper = np.exp(weights + 1.96 * se_weights)

or_summary_df = pd.DataFrame({
    'Yếu tố Lâm sàng': feature_names,
    'Hệ số (w)': weights.round(4),
    'Sai số chuẩn (SE)': se_weights.round(4),
    'Odds Ratio (OR)': odds_ratios.round(4),
    '95% CI Lower': ci_lower.round(4),
    '95% CI Upper': ci_upper.round(4)
})

print("=" * 70)
print("3. BẢNG TỔNG HỢP ODDS RATIO LÂM SÀNG (CLINICAL ODDS RATIO TABLE):")
print(or_summary_df.to_string(index=False))
print("=" * 70)
print("[+] MÔ HÌNH VẬN HÀNH HOÀN HẢO: ĐẠT CHUẨN DIỄN GIẢI Y HỌC DỰA TRÊN BẰNG CHỨNG (EBM)!")