# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 2 Code Lab: Xây dựng Data Pipeline Y sinh Hoàn chỉnh Chống Rò rỉ
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi: EDA, Xử lý Missing Data đa cơ chế, Outlier Filtering, Pipeline Chống Rò rỉ.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------------------------------------
# 1. KHỞI TẠO TẬP DỮ LIỆU LÂM SÀNG MÔ PHỎNG (500 BỆNH NHÂN CẤP CỨU)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 500

# 1.1 Đặc trưng nhân khẩu học & Sinh hiệu
age = np.random.normal(loc=62, scale=12, size=n_samples).clip(18, 95)
gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45])
sbp = np.random.normal(loc=135, scale=20, size=n_samples) # Huyết áp tâm thu

# 1.2 Xét nghiệm Sinh hóa có phân phối lệch & Ngoại lai
glucose = np.random.lognormal(mean=4.8, sigma=0.4, size=n_samples) # Glucose (mg/dL)
alt = np.random.exponential(scale=35, size=n_samples) + 15 # Men gan ALT (U/L)
troponin = np.random.exponential(scale=0.08, size=n_samples) # Men tim (ng/mL)

# 1.3 Tạo nhãn đích Bệnh tim cấp tính (Acute Coronary Syndrome)
prob = 1 / (1 + np.exp(-(0.04*age + 0.02*sbp + 12*troponin + 0.01*glucose - 7.5)))
y = np.random.binomial(1, prob, size=n_samples)

# 1.4 Gây khuyết thiếu dữ liệu theo cơ chế thực tế
# - Glucose: MCAR (mất ngẫu nhiên 8% do lỗi truyền dữ liệu)
glucose_mask = np.random.rand(n_samples) < 0.08
glucose[glucose_mask] = np.nan

# - Troponin: MAR (Bác sĩ chỉ định đo Troponin phụ thuộc vào Huyết áp và Tuổi)
mar_prob = 1 / (1 + np.exp(-(0.05*age + 0.02*sbp - 6)))
troponin_mask = np.random.rand(n_samples) > mar_prob # Người trẻ ít đo hơn
troponin[troponin_mask] = np.nan

# 1.5 Cài đặt giá trị ngoại lai sinh lý bất khả thi (lỗi nhập liệu)
sbp[10] = 999.0 # Lỗi gõ phím huyết áp
alt[25] = 4500.0 # Bệnh nhân viêm gan tối cấp (Ngoại lai bệnh lý thực)

df = pd.DataFrame({
    'patient_id': [f"PT_{i:04d}" for i in range(n_samples)],
    'age': age,
    'gender': gender,
    'sbp': sbp,
    'glucose': glucose,
    'alt': alt,
    'troponin': troponin,
    'target': y
})

print("=" * 70)
print("1. TỔNG QUAN DỮ LIỆU BỆNH ÁN ĐIỆN TỬ BAN ĐẦU (EHR DATASET):")
print(df.head())
print("\nThống kê tỷ lệ dữ liệu thiếu (%):")
print(df.isnull().mean() * 100)
print("=" * 70)

# ---------------------------------------------------------
# 2. LÀM SẠCH NGOẠI LAI SINH LÝ BẤT KHẢ THI (RULE-BASED CLEANING)
# ---------------------------------------------------------
# Huyết áp tâm thu sbp > 300 mmHg là bất khả thi về sinh học -> Gán thành NaN
df.loc[df['sbp'] > 300, 'sbp'] = np.nan

# ---------------------------------------------------------
# 3. PHÂN CHIA DỮ LIỆU THEO BỆNH NHÂN (TRAIN / TEST SPLIT)
# ---------------------------------------------------------
X = df[['age', 'gender', 'sbp', 'glucose', 'alt', 'troponin']]
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 4. XÂY DỰNG DATA PIPELINE HOÀN CHỈNH BẰNG SCIKIT-LEARN
# ---------------------------------------------------------
# Định nghĩa danh sách các nhóm cột
skewed_features = ['alt', 'glucose', 'troponin'] # Cần RobustScaler & KNN Imputer
normal_features = ['age', 'sbp']                 # Cần StandardScaler & SimpleImputer
categorical_features = ['gender']                # Cần OneHotEncoder

# Pipeline cho biến phân phối lệch (Robust)
skewed_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

# Pipeline cho biến chuẩn (Gaussian)
normal_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline cho biến phân loại
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Hợp nhất qua ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('skewed', skewed_pipeline, skewed_features),
    ('normal', normal_pipeline, normal_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Pipeline hoàn chỉnh đóng gói cả Tiền xử lý và Mô hình
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
])

# ---------------------------------------------------------
# 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ (FIT DUY NHẤT TRÊN TRAIN)
# ---------------------------------------------------------
full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
y_prob = full_pipeline.predict_proba(X_test)[:, 1]

print("\n2. KẾT QUẢ ĐÁNH GIÁ PIPELINE TRÊN TẬP TEST ĐỘC LẬP (N=125):")
print(f"  - ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
print(classification_report(y_test, y_pred, target_names=['Không biến cố', 'Hội chứng vành cấp']))
print("=" * 70)
print("[+] PIPELINE THỰC THI THÀNH CÔNG: KHÔNG RÒ RỈ DỮ LIỆU, XỬ LÝ ĐA PHƯƠNG THỨC!")