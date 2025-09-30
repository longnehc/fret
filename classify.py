import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 读取数据
truth = np.load('truth.npy')    # shape (500, 500)
smooth = np.load('smooth.npy')  # shape (500, 500)

# 2. 取出“第一组”数据
truth_grp0 = truth[0, :].reshape(-1, 1)    # (500, 1)
smooth_grp0 = smooth[0, :].reshape(-1, 1)  # (500, 1)

# 3. 在 K=2…10 范围内，用轮廓系数选最优 K
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(smooth_grp0)
    sil_scores.append(silhouette_score(smooth_grp0, labels))

# 4. 绘制轮廓系数曲线，帮助确认
plt.figure()
plt.plot(list(K_range), sil_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Group 0')
plt.show()

# 5. 选出最高分对应的 K
best_k = K_range[int(np.argmax(sil_scores))]
print(f"Optimal K for group 0: {best_k}")

# 6. 用 best_k 做 K-Means 聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(smooth_grp0)

# 7. 将真值映射到整数标签（如果真值本身不是 0..K-1）
unique_vals = np.unique(truth_grp0)
truth_label_map = {v: i for i, v in enumerate(unique_vals)}
truth_labels = np.vectorize(truth_label_map.get)(truth_grp0.flatten())

print("Truth unique values (ideal centers):")
print(unique_vals)

# ————— 在这里插入对 kmeans.cluster_centers_ 的合并操作 —————
# 8. 找出需要合并的中心：任意两中心差值 < 0.1
orig_centers = kmeans.cluster_centers_.flatten()
print("\nOriginal K-Means cluster centers:")
print(orig_centers)

# 8. 找出需要合并的中心：任意两中心差值 < 0.1 且 round(ci,1) 相同
# 初始合并
merged = []
used = set()
threshold = 0.1

# 初次合并的逻辑
for i, ci in enumerate(orig_centers):
    if i in used:
        continue
    group = [ci]
    used.add(i)
    for j, cj in enumerate(orig_centers):
        if j not in used:
            same_first_decimal = int(ci * 10) == int(cj * 10)  # 判断小数点后第一位是否相同
            if abs(ci - cj) < threshold and same_first_decimal:
                group.append(cj)
                used.add(j)
    merged_center = np.mean(group)
    merged.append(merged_center)

# 初步合并后的中心
merged_centers = np.array(sorted(merged))
print("\nInitial merged cluster centers (Δ<0.1 且小数点后第一位相同):")
print(merged_centers)

# 继续合并：对于新合并的中心继续按相同规则合并
def merge_centers(centers, threshold=0.1):
    merged = []
    used = set()

    for i, ci in enumerate(centers):
        if i in used:
            continue
        group = [ci]
        used.add(i)
        for j, cj in enumerate(centers):
            if j not in used:
                # 比较小数点后第一位
                same_first_decimal = int(ci * 10) == int(cj * 10)
                if abs(ci - cj) < threshold and same_first_decimal:
                    group.append(cj)
                    used.add(j)
        merged_center = np.mean(group)
        merged.append(merged_center)

    return np.array(sorted(merged))

# 循环直到不再需要合并
while True:
    new_merged_centers = merge_centers(merged_centers, threshold)
    if len(new_merged_centers) == len(merged_centers):
        break
    merged_centers = new_merged_centers

print("\nFinal merged cluster centers after all merges:")
print(merged_centers)


# 9. 将原始标签映射到合并后标签
# 首先，为每个原始中心找到它对应的合并中心索引
center_map = {}
for orig in orig_centers:
    # 找到最接近的 merged_center
    idx = np.argmin(np.abs(merged_centers - orig))
    center_map[orig] = idx

# 构建新的标签
cluster_labels_merged = np.array([center_map[c] for c in kmeans.cluster_centers_.flatten()[cluster_labels]])

# 10. 并排可视化真值 vs 原始聚类 vs 合并后聚类
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

# 真值
axes[0].scatter(range(500), truth_labels, c=truth_labels, cmap='tab10', s=20)
axes[0].set_title('Truth Labels (Group 0)')
axes[0].set_xlabel('Point Index')
axes[0].set_ylabel('Class Label')

# 原始 K-Means
axes[1].scatter(range(500), cluster_labels, c=cluster_labels, cmap='tab10', s=20)
axes[1].set_title(f'K-Means Clusters (K={best_k})')
axes[1].set_xlabel('Point Index')

# 合并后
axes[2].scatter(range(500), cluster_labels_merged, c=cluster_labels_merged, cmap='tab10', s=20)
axes[2].set_title('Merged Clusters (Δ<0.1)')
axes[2].set_xlabel('Point Index')

plt.tight_layout()
plt.show()
