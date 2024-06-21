"""
加载训练集、验证集以及测试集
"""
import os, json, pickle
import numpy as np


# 根据sc_type获取不同智能合约漏洞数据集的地址
def get_data_path_vocab(sc_type):
    if sc_type == "RE":
        train_data_path = "Datasets/RE/training_data.json"
        valid_data_path = "Datasets/RE/valid_data.json"
        test_data_path = "Datasets/RE/test_data.json"
        vocab2id_path = "Datasets/RE/RE_vocab_id.pkl"
    elif sc_type == "TD":
        train_data_path = "Datasets/TD/training_data.json"
        valid_data_path = "Datasets/TD/valid_data.json"
        test_data_path = "Datasets/TD/test_data.json"
        vocab2id_path = "Datasets/TD/TD_vocab_id.pkl"
    elif sc_type == "IOU":
        train_data_path = "Datasets/IOU/training_data.json"
        valid_data_path = "Datasets/IOU/valid_data.json"
        test_data_path = "Datasets/TD/test_data.json"
        vocab2id_path = "Datasets/IOU/IOU_vocab_id.pkl"
    return train_data_path, valid_data_path, test_data_path, vocab2id_path



# 从SC数据集加载数据；使用max_setence_length进行截断或补全，来确保路径长度一致
def get_SC_data(sc_json_path, max_setence_length):
    # 初始化合约数据及标签的列表
    sc_data, sc_label = [], []

    # 读取所有数据
    with open(sc_json_path, 'r', encoding='utf-8') as file:
        # 使用json.load()方法解析JSON数据
        all_data = file.readlines()

    for i in all_data:
        if i.strip() == "":
            continue
        one_sol_json = json.loads(i)

        sol_content = one_sol_json["sol content"].split(" ")
        # 计算sol_content的长度，判断是否需要进行截断
        if len(sol_content) < max_setence_length:
            sol_content.extend(["PAD"]*(max_setence_length-len(sol_content)))
        else:
            sol_content = sol_content[:max_setence_length]

        sc_data.append(sol_content)
        sc_label.append(one_sol_json["label"])

    # print("The length of train data is {0}, and max_setence_length is {1}".format(len(op_data), max_setence_length))
    return np.array(sc_data), np.array(sc_label)


# 加载本地训练数据
def load_train_valid_test_data(SC_Type, max_setence_length):

    train_data_path, valid_data_path, test_data_path, vocab2id_path = get_data_path_vocab(SC_Type)

    # 加载词汇表
    with open(vocab2id_path, 'rb') as f:
        vocab2id = pickle.load(f)

    # 加载训练数据，并且ont-hot向量化，对应word2vec词嵌入
    temp_data, train_label = get_SC_data(train_data_path, max_setence_length)
    train_data = []
    for i in temp_data:
        temp = []
        for j in i:
            if j in vocab2id:
                temp.append(vocab2id[j])
            else:
                temp.append(vocab2id["PAD"])
        train_data.append(temp)

    # 加载验证数据，并且ont-hot向量化，对应word2vec词嵌入
    temp_data, valid_label = get_SC_data(valid_data_path, max_setence_length)
    valid_data = []
    for i in temp_data:
        temp = []
        for j in i:
            if j in vocab2id:
                temp.append(vocab2id[j])
            else:
                temp.append(vocab2id["PAD"])
        valid_data.append(temp)

    # 加载测试数据，并且ont-hot向量化，对应word2vec词嵌入
    temp_data, test_label = get_SC_data(test_data_path, max_setence_length)
    test_data = []
    for i in temp_data:
        temp = []
        for j in i:
            if j in vocab2id:
                temp.append(vocab2id[j])
            else:
                temp.append(vocab2id["PAD"])
        test_data.append(temp)

    return np.array(train_data), train_label, np.array(valid_data), valid_label, np.array(test_data), test_label


# 对训练集数据的标签进行加噪
def get_training_data_with_noisy_labels(train_data, train_label, is_symmetric=False, mislabel_rate=0.3, num_classes=2):
    train_data_clean, train_data_noise, train_label_clean, train_label_noise = [], [], [], []
    for i in range(num_classes):
        indices = np.where(np.array(train_label) == i)[0]
        np.random.shuffle(indices)

        noisy_data_num = int(len(indices) * mislabel_rate)
        noisy_data_idx, clean_data_idx = indices[:noisy_data_num], indices[noisy_data_num: ]
        # 读写有噪声的训练集
        for j in noisy_data_idx:
            if is_symmetric == False:
                train_data_noise.append(train_data[j])
                train_label_noise.append(train_label[j])
            else:
                train_data_noise.append(train_data[j])
                train_label_noise.append(num_classes - train_label[j] -1)

        # 读写干净的训练集
        for j in clean_data_idx:
            train_data_clean.append(train_data[j])
            train_label_clean.append(train_label[j])

    return np.array(train_data_clean), np.array(train_data_noise), np.array(train_label_clean), np.array(train_label_noise)


if __name__ == "__main__":
    SC_Type, max_setence_length = "RE", 2000
    load_train_valid_test_data(SC_Type, max_setence_length)


