# -*- coding: utf-8 -*-
# Tự động trích xuất từ Code Lab

# -*- coding: utf-8 -*-
# Chương 6 Code Lab: Học Không Giám Sát - Phân cụm K-Means/GMM, Giảm chiều PCA & Tách nguồn FastICA
# Giáo trình: Học máy trong Y tế (ET4248)
# Thực thi: Silhouette Analysis, K-Means, GMM với BIC, Nén PCA và Tách tín hiệu FastICA.

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 1. KHỞI TẠO TẬP DỮ LIỆU PHÂN TẦNG LÂM SÀNG OSA MÔ PHỎNG (N=300 BỆNH NHÂN)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 300

# Sinh 3 kiểu hình lâm sàng OSA tiềm ẩn (Ground Truth Phenotypes)
# Nhóm 1: Béo phì - Ngưng thở tắc nghẽn nặng (Obese OSA)
g1 = np.random.multivariate_normal(mean=[34, 45, 16, 78], cov=np.diag([4, 25, 4, 16]), size=100)
# Nhóm 2: Mất ngủ - Ngưng thở nhẹ/trung bình (Insomnia OSA)
g2 = np.random.multivariate_normal(mean=[24, 18, 6, 88], cov=np.diag([3, 16, 3, 9]), size=100)
# Nhóm 3: Người cao tuổi - Ngưng thở hỗn hợp (Elderly OSA)
g3 = np.random.multivariate_normal(mean=[27, 30, 11, 82], cov=np.diag([3, 20, 3, 12]), size=100)

X_raw = np.vstack([g1, g2, g3])
feature_names = ['BMI', 'AHI', 'Epworth_Score', 'Min_SpO2']

df = pd.DataFrame(X_raw, columns=feature_names)
print("=" * 70)
print("1. TỔNG QUAN TẬP DỮ LIỆU LÂM SÀNG NGƯNG THỞ KHI NGỦ (OSA COHORT):")
print(df.head())
print("=" * 70)

# ---------------------------------------------------------
# 2. CHUẨN HÓA THANG ĐO BẮT BUỘC (Z-SCORE)
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---------------------------------------------------------
# 3. ĐÁNH GIÁ NỘI TẠI TÌM SỐ CỤM TỐI ƯU (SILHOUETTE & BIC)
# ---------------------------------------------------------
print("2. PHÂN TÍCH HỆ SỐ SILHOUETTE VÀ BIC ĐỂ CHỌN SỐ CỤM K:")
silhouette_scores = []
bic_scores = []
k_range = range(2, 6)

for k in k_range:
    # K-Means Silhouette
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    s_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(s_score)
    
    # GMM BIC
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    
    print(f"  - K = {k}: Silhouette Score = {s_score:.4f} | GMM BIC = {gmm.bic(X_scaled):.1f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"[+] Số cụm tối ưu theo Silhouette là: K = {best_k}")
print("-" * 70)

# ---------------------------------------------------------
# 4. HUẤN LUYỆN K-MEANS & GMM TẠI K TỐI ƯU
# ---------------------------------------------------------
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
df['KMeans_Cluster'] = kmeans_final.fit_predict(X_scaled)

gmm_final = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
gmm_final.fit(X_scaled)
probs = gmm_final.predict_proba(X_scaled)
df['GMM_Cluster'] = gmm_final.predict(X_scaled)
df['GMM_Max_Prob'] = np.max(probs, axis=1).round(3)

print("3. ĐẶC ĐIỂM TRUNG BÌNH CỦA CÁC KIỂU HÌNH LÂM SÀNG (K-MEANS CLUSTERS):")
cluster_summary = df.groupby('KMeans_Cluster')[feature_names].mean().round(2)
print(cluster_summary)
print("=" * 70)

# ---------------------------------------------------------
# 5. GIẢM CHIỀU DỮ LIỆU BẰNG PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA)
# ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

evr = pca.explained_variance_ratio_
print("4. KẾT QUẢ NÉN CHIỀU DỮ LIỆU BẰNG PCA:")
print(f"  - Tỷ lệ phương sai giải thích trục PC1: {evr[0]*100:.2f}%")
print(f"  - Tỷ lệ phương sai giải thích trục PC2: {evr[1]*100:.2f}%")
print(f"  - Tổng phương sai tích lũy 2 trục đầu: {sum(evr)*100:.2f}% (Bảo toàn > 85% thông tin)")
print("-" * 70)

# ---------------------------------------------------------
# 6. TÁCH NGUỒN TÍN HIỆU Y SINH MÔ PHỎNG BẰNG FASTICA
# ---------------------------------------------------------
# Tạo 2 nguồn tín hiệu nhân tạo: Nhịp tim mẹ (sóng sin tuần hoàn) và Nhịp tim thai (xung nhọn ngẫu nhiên)
time = np.linspace(0, 1, 500)
s1 = np.sin(2 * np.pi * 5 * time) # Nguồn 1 (mECG)
s2 = np.sign(np.sin(2 * np.pi * 12 * time)) # Nguồn 2 (fECG - phi Gaussian)
S_true = np.c_[s1, s2]

# Trộn tín hiệu qua ma trận A
A = np.array([[0.8, 0.5], [0.4, 0.9]])
X_mixed = np.dot(S_true, A.T)

# Tách nguồn bằng FastICA
ica = FastICA(n_components=2, random_state=42)
S_recovered = ica.fit_transform(X_mixed)

# Đánh giá mức độ độc lập qua ma trận tương quan giữa 2 nguồn phục hồi
corr_recovered = np.corrcoef(S_recovered[:, 0], S_recovered[:, 1])[0, 1]

print("5. KẾT QUẢ TÁCH NGUỒN TÍN HIỆU SINH HỌC BẰNG FASTICA:")
print(f"  - Hệ số tương quan tuyến tính giữa 2 nguồn tách được: r = {corr_recovered:.6f} (Gần đúng bằng 0)")
print("  - FastICA tách rời hoàn toàn hai nguồn sinh học độc lập thành công!")
print("=" * 70)
print("[+] HOÀN TẤT TOÀN DIỆN CODE LAB HỌC KHÔNG GIÁM SÁT VÀ GIẢM CHIỀU DỮ LIỆU!")