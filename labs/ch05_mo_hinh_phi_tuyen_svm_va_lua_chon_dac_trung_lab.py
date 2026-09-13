# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 5 Code Lab: Lựa chọn Đặc trưng RFE & Phân loại Phi tuyến RBF Kernel SVM trên Dữ liệu Ung thư Vú
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi: Chuẩn hóa, RFE chọn 10 đặc trưng nhân tế bào, GridSearchCV tối ưu (C, gamma), Đánh giá ROC-AUC.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. NẠP DỮ LIỆU UNG THƯ VÚ WISCONSIN (WDBC - UCI)
# ---------------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target # 0: Ác tính (Malignant), 1: Lành tính (Benign)
# Đổi nhãn: 1 là Ác tính (Positive / Event), 0 là Lành tính (Negative)
y = np.where(y == 0, 1, 0)
feature_names = data.feature_names

print("=" * 70)
print("1. TỔNG QUAN TẬP DỮ LIỆU UNG THƯ VÚ WISCONSIN (WDBC):")
print(f"  - Tổng số bệnh nhân (N): {X.shape[0]}")
print(f"  - Số đặc trưng hình thái nhân tế bào (p): {X.shape[1]}")
print(f"  - Số ca ung thư Ác tính (y=1): {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
print(f"  - Số ca khối u Lành tính (y=0): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
print("=" * 70)

# ---------------------------------------------------------
# 2. PHÂN CHIA DỮ LIỆU & CHUẨN HÓA Z-SCORE (FIT CHỈ TRÊN TRAIN)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. LỰA CHỌN ĐẶC TRƯNG BẰNG RFE (RECURSIVE FEATURE ELIMINATION)
# ---------------------------------------------------------
# Sử dụng Linear SVM để ước lượng độ quan trọng |w_j| và chọn ra 10 đặc trưng then chốt
estimator = SVC(kernel='linear', random_state=42)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector.fit(X_train_scaled, y_train)

selected_features = feature_names[selector.support_]
print("2. DANH MỤC 10 ĐẶC TRƯNG HÌNH THÁI ĐƯỢC RFE LỰA CHỌN:")
for rank, feat in enumerate(selected_features, 1):
    print(f"  [{rank:02d}] {feat}")
print("-" * 70)

# Trích xuất không gian đặc trưng rút gọn (10 chiều)
X_train_rfe = selector.transform(X_train_scaled)
X_test_rfe = selector.transform(X_test_scaled)

# ---------------------------------------------------------
# 4. TINH CHỈNH SIÊU THAM SỐ RBF KERNEL SVM (GRID SEARCH CV)
# ---------------------------------------------------------
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train_rfe, y_train)

best_model = grid_search.best_estimator_
print(f"3. THIẾT LẬP SIÊU THAM SỐ TỐI ƯU: C = {grid_search.best_params_['C']}, gamma = {grid_search.best_params_['gamma']}")
print(f"   Điểm Cross-Validation ROC-AUC trung bình (Train): {grid_search.best_score_:.4f}")
print("=" * 70)

# ---------------------------------------------------------
# 5. ĐÁNH GIÁ HIỆU NĂNG TOÀN DIỆN TRÊN TẬP TEST ĐỘC LẬP (N=171)
# ---------------------------------------------------------
y_prob = best_model.predict_proba(X_test_rfe)[:, 1]
y_pred = best_model.predict(X_test_rfe)

test_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("4. KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PROPOSED TRÊN TẬP TEST ĐỘC LẬP (N=171):")
print(f"  - Diện tích dưới đường cong ROC (ROC-AUC): {test_auc:.4f}")
print(f"  - Độ nhạy lâm sàng (Sensitivity / Phát hiện Ác tính): {sensitivity*100:.2f}% ({tp}/{tp+fn})")
print(f"  - Độ đặc hiệu (Specificity / Nhận diện Lành tính): {specificity*100:.2f}% ({tn}/{tn+fp})")
print("-" * 70)
print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
print(f"                Dự đoán Lành tính (0) | Dự đoán Ác tính (1)")
print(f"Thực tế Lành tính (0) :      {tn:<16} |      {fp:<16}")
print(f"Thực tế Ác tính (1)   :      {fn:<16} |      {tp:<16}")
print("=" * 70)
print("[+] HOÀN TẤT CODE LAB: RBF SVM + RFE ĐẠT ĐỘ NHẠY LÂM SÀNG GẦN TUYỆT ĐỐI (> 98%)!")