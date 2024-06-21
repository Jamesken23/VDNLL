from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def select_clean_samples(features, X_train, Y_train, forget_rate=0.2):

    # 配置KNN算法的相关信息
    knn_k = int((len(X_train)/2) ** 0.5)
    knn = KNeighborsClassifier(n_neighbors=knn_k)
    knn.fit(features, Y_train)

    distances_sum, indexes_knn = knn.kneighbors(features, knn_k, True)
    distances_sum = distances_sum.sum(axis=1)

    # 每个epoch中需要使用的干净数据集的比例
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(X_train))

    # 计算每个样本与邻居样本之间的特征表示距离，取平均值
    train_x_means = []
    for indexes in indexes_knn:
        indexes_1d = (np.array(indexes).flatten()).tolist()
        train_x_near = features[indexes_1d]
        train_x_near_mean = train_x_near.mean(axis=0)
        train_x_means.append(train_x_near_mean)
    distances_mean = np.linalg.norm(features - train_x_means, axis=1, ord=1)

    # 分别计算standard_mean和standard_sum
    standards = sorted(distances_mean)
    standard_mean = standards[num_remember]

    standards = sorted(distances_sum)
    standard_sum = standards[num_remember]

    # 获得干净的数据集
    clean_x, clean_y = [], []
    for index in range(len(X_train)):
        if abs(distances_mean[index]-standard_mean) < 0.5 and abs(distances_sum[index]-standard_sum) < 0.5:
            clean_x.append(X_train[index])
            # 更新标签信息
            x_near_y_idx = (np.array(indexes_knn[index]).flatten()).tolist()
            near_y = Y_train[x_near_y_idx]
            # 多数投票法来确定标签
            if sum(near_y) > int(len(near_y) / 2 + 0.0001):
                clean_y.append(1)
            else:
                clean_y.append(0)
        else:
            if distances_mean[index] < standard_mean and distances_sum[index] < standard_sum:
                clean_x.append(X_train[index])
                # 更新标签信息
                x_near_y_idx = (np.array(indexes_knn[index]).flatten()).tolist()
                near_y = Y_train[x_near_y_idx]
                # 多数投票法来确定标签
                if sum(near_y) > int(len(near_y)/2 + 0.0001):
                    clean_y.append(1)
                else:
                    clean_y.append(0)
    return np.array(clean_x), np.array(clean_y)