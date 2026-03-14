import numpy as np

def kmeans(data, k, thresh=1, max_iterations=100):
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    labels = np.zeros(data.shape[0], dtype=int)

    for iteration in range(max_iterations):
        # numpy广播机制
        # 在K-means算法中：
        # data[:, None] 从 (n, f) 变为 (n, 1, f)，添加了一个维度
        # centers 是 (k, f)
        # 广播后变成 (n, k, f)，表示每个数据点与每个中心在每个特征上的差值
        distances = np.linalg.norm(data[:, None] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        center_change = np.linalg.norm(centers - new_centers)
        if center_change < thresh:
            break
        
        centers = new_centers

    return labels, centers

if __name__ == "__main__":
    data = np.random.rand(100, 2)
    k = 3
    labels, centers = kmeans(data, k)
    print("current labels:", labels)
    print("current centers:", centers)