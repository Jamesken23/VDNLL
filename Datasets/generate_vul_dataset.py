"""
从原始的漏洞数据集中分别加载出三个子集，并且拆分成训练集、验证集、测试集
原始数据集的来源：https://github.com/Messi-Q/Smart-Contract-Dataset
"""
import os, json
import numpy as np


def load_vul_dataset(original_vul_sol_path, new_vul_sol_path):

    for i in os.listdir(original_vul_sol_path):
        if ".sol" not in i:
            continue

        sol_file_path = os.path.join(original_vul_sol_path, i)
        # 创建空列表
        text = ""
        # 打开合约文件
        with open(sol_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            # 读取全部内容 ，并以列表方式返回
            lines = file.readlines()
        for line in lines:
            # 如果读到空行，就跳过
            if line.isspace():
                continue
            else:
                text += line.strip() + " "
        # print("sol_file_path is {0}".format(text))
        idx_label = i.split(".")[0]
        sol_label = int(idx_label.split("_")[-1])

        one_sol_dict = {}
        one_sol_dict["sol name"] = i
        one_sol_dict["label"] = sol_label
        one_sol_dict["sol content"] = text

        #  保存数据,每个合约数据以换行结束
        with open(new_vul_sol_path, 'a') as ff:
            json.dump(one_sol_dict, ff)
            ff.write('\n')
        print("We have write {0}".format(one_sol_dict["sol name"]))


# 将所有数据拆分成训练集、验证集、测试集
def split_all_sol(new_vul_sol_path, train_path, valid_path, test_path, training_data_ratio, valid_data_ratio,
                  num_classes=2):
    # 读取源文件中的合约数据
    with open(new_vul_sol_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    all_sol_name, all_data, all_labels = [], [], []
    for i in all_lines:
        temp = json.loads(i)
        all_sol_name.append(temp["sol name"])
        all_data.append(temp["sol content"])
        all_labels.append(temp["label"])

    data_x, label_x, data_u, label_u = [], [], [], []
    for i in range(num_classes):
        indices = np.where(np.array(all_labels) == i)[0]
        np.random.shuffle(indices)

        training_num = int(len(indices) * training_data_ratio)
        valid_num = int(len(indices) * (training_data_ratio + valid_data_ratio))
        training_idx, valid_idx, test_idx = indices[:training_num], indices[training_num: valid_num], indices[valid_num:]

        # 读写训练集
        for j in training_idx:
            one_sol_dict = {}
            one_sol_dict["sol name"] = all_sol_name[j]
            one_sol_dict["label"] = all_labels[j]
            one_sol_dict["sol content"] = all_data[j]

            #  保存数据,每个合约数据以换行结束
            with open(train_path, 'a') as ff:
                json.dump(one_sol_dict, ff)
                ff.write('\n')

        # 读写验证集
        for j in valid_idx:
            one_sol_dict = {}
            one_sol_dict["sol name"] = all_sol_name[j]
            one_sol_dict["label"] = all_labels[j]
            one_sol_dict["sol content"] = all_data[j]

            #  保存数据,每个合约数据以换行结束
            with open(valid_path, 'a') as ff:
                json.dump(one_sol_dict, ff)
                ff.write('\n')

        # 读写测试集
        for j in test_idx:
            one_sol_dict = {}
            one_sol_dict["sol name"] = all_sol_name[j]
            one_sol_dict["label"] = all_labels[j]
            one_sol_dict["sol content"] = all_data[j]

            #  保存数据,每个合约数据以换行结束
            with open(test_path, 'a') as ff:
                json.dump(one_sol_dict, ff)
                ff.write('\n')



if __name__ == "__main__":
    original_vul_sol_path = r"original_sol/DE_414_0_278_1_136"
    new_vul_sol_path = r"DE/all_sol.json"
    load_vul_dataset(original_vul_sol_path, new_vul_sol_path)

    # 训练集、验证机、测试集的比例分别为8：1：1
    training_data_ratio, valid_data_ratio, test_data_ratio = 0.8, 0.1, 0.1
    train_path = r"DE/training_data.json"
    valid_path = r"DE/valid_data.json"
    test_path = r"DE/test_data.json"
    split_all_sol(new_vul_sol_path, train_path, valid_path, test_path, training_data_ratio, valid_data_ratio,
                  num_classes=2)