# coding='utf-8'
"""t-SNE对特征向量进行可视化"""
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_feature_list(json_data=r"D:\Downloads\VDNLL_Self_Att_SU_noise_1_epoch_20.json"):
    with open(json_data, 'r') as f:
        data = json.load(f)
    # data_json = {"Features": feature_numpy, "Predictions": pred_numpy, "True_labels": label_numpy}
    features, preds, labels = data["Features"], data["Predictions"], data["True_labels"]
    return np.array(features), np.array(labels)


# 对输入数据进行归一化处理
def is_normalization(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    data = (X - x_min) / (x_max - x_min)
    return data

def plot_embedding(data, label, title):
    data = is_normalization(data)

    flag_1, flag_2 = False, False

    fig = plt.figure()
    ax = plt.subplot(111)

    flag_xy = []

    for i in range(data.shape[0]):
        if [data[i, 0], data[i, 1]] in flag_xy:
            continue

        if label[i] == 0:
            if flag_1 is False:
                plt.scatter(data[i, 0], data[i, 1], marker='o', c='r', label="non-bug")
                # plt.scatter(data[i, 0], data[i, 1], marker='o', c='r')
                flag_1 = True
            else:
                plt.scatter(data[i, 0], data[i, 1], marker='o', c='r')
        else:
            if flag_2 is False:
                plt.scatter(data[i, 0], data[i, 1], marker='x', c='b', label="bug")
                # plt.scatter(data[i, 0], data[i, 1], marker='x', c='b')
                flag_2 = True
            else:
                plt.scatter(data[i, 0], data[i, 1], marker='x', c='b')

        flag_xy.append([data[i, 0], data[i, 1]])
    # plt.title(title)
    return fig


if __name__ == '__main__':
    json_data = r"D:\JamesFiles\Latex Project\2024_VDNLL\2024_TSE_VDNLL\Data\T_NSE\VDNLL\VDNLL_Self_Att_RE_noise_10_epoch_10.json"
    features, labels = get_feature_list(json_data)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(features)
    print('result.shape',result.shape)
    plt.rcParams.update({"font.size": 16})
    fig = plot_embedding(result, labels, "T-NSE")
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.legend()
    plt.show()