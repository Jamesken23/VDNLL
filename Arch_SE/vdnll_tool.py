from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cal_cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom



# 定义softmax函数
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# 利用numpy计算交叉熵损失，对应pytorch的 F.cross_entropy(y_1, t, reduce=False)
# def calculate_cross_entropy(x, y):
#     x_softmax = [softmax(x[i]) for i in range(len(x))]
#     x_log = [np.log(x[i] * y[i] + 0.001) for i in range(len(y))]
#     # loss = - np.sum(x_log) / len(y)
#     return -1 * x_log


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = abs(data - mean) / std_dev
    return normalized_data


def select_clean_samples(features_1, logits_2, X_train, Y_train, t1=0.7, t2=3.5):

    # 配置KNN算法的相关信息
    knn_k = int((len(X_train) / 2) ** 0.5)
    knn = KNeighborsClassifier(n_neighbors=knn_k)
    knn.fit(features_1, Y_train)

    distances_sum, indexes_knn = knn.kneighbors(features_1, knn_k, True)

    # 计算每个样本与邻居样本之间的特征表示距离，取平均值
    train_x_means = []
    for indexes in indexes_knn:
        indexes_1d = (np.array(indexes).flatten()).tolist()
        train_x_near = features_1[indexes_1d]
        train_x_near_mean = train_x_near.mean(axis=0)
        train_x_means.append(train_x_near_mean)

    all_similarity = []
    for index in range(len(X_train)):
        cos_sim = cal_cosine_similarity(features_1[index], train_x_means[index])
        all_similarity.append(cos_sim)

    # print("all_similarity shape is {0}".format(np.array(all_similarity).shape))
    all_similarity_sum = sum(all_similarity) + 0.0001

    # 选择model_1对应的干净数据集
    clean_x_1, clean_y_1 = [], []
    for index in range(len(X_train)):
        if 1.0 - float(all_similarity[index])/all_similarity_sum > t1:
            clean_x_1.append(X_train[index])
            clean_y_1.append(Y_train[index])
        else:
            clean_x_1.append(X_train[index])
            clean_y_1.append(1 - Y_train[index])

    # print("the shape of clean_x_1 is {0}".format(np.array(clean_x_1).shape))

    # 计算model_2的训练损失
    logits_2_tensor, Y_train_tensor = torch.FloatTensor(logits_2), torch.LongTensor(Y_train)
    ce_loss = F.cross_entropy(logits_2_tensor, Y_train_tensor, reduce=False)

    ce_loss = ce_loss.data.tolist()
    # print("the shape of ce_loss is {0}".format(np.array(ce_loss).shape))
    loss_z_score = z_score_normalize(np.array(ce_loss))
    # print("the shape of loss_z_score is {0}".format(np.array(loss_z_score).shape))

    # 选择model_2对应的干净数据集
    clean_x_2, clean_y_2 = [], []
    for index in range(len(X_train)):
        if loss_z_score[index] < t2:
            clean_x_2.append(X_train[index])
            clean_y_2.append(Y_train[index])
        else:
            clean_x_2.append(X_train[index])
            clean_y_2.append(1 - Y_train[index])

    # print("the shape of clean_x_2 is {0}".format(np.array(clean_x_2).shape))

    # 确定最终的干净数据集及标签
    clean_x, clean_y = [], []
    for index in range(len(X_train)):
        if clean_y_1[index] == clean_y_2[index]:
            clean_x.append(X_train[index])
            clean_y.append(clean_y_1[index])
        else:
            clean_x.append(X_train[index])
            clean_y.append(clean_y_1[index] + clean_y_2[index])

    # print("the shape of clean_x is {0}".format(np.array(clean_x).shape))
    return np.array(clean_x), np.array(clean_y)


if __name__ == '__main__':
    data = [0.11, 1.33, 1.67, 4.83, 6.23]
    normalized_data = z_score_normalize(np.array(data))
    print(normalized_data)