import os
import pickle, json


# 读取某个漏洞数据集下的所有数据
def get_all_data(sc_data_path):
    all_sol = []
    # 首先读取所有数据
    with open(sc_data_path, 'r', encoding='utf-8') as file:
        # 使用json.load()方法解析JSON数据
        all_data = file.readlines()

    for i in all_data:
        if i.strip() == "":
            continue
        one_sol_json = json.loads(i)
        long_path_list = one_sol_json["sol content"].split(" ")
        all_sol.extend(long_path_list)

    return all_sol


# 获取词汇表，用于one-hot向量
def word_process(sc_data_path, sc_type):
    word_list = []

    all_sol_data = get_all_data(sc_data_path)
    # print("all_sol_data is: {0}".format(all_sol_data))
    # 清除一些噪声字符
    for i in all_sol_data:
        if i in ['{', '}', '(', ')', '[', ']', ' ', ';', ',', '//', '.', '**', '!', '/*', '*/']:
            continue
        # if '(' in i or ')' in i or '{' in i or '}' in i or ';' in i or ',' in i or '[' in i or ']' in i:
        #     continue
        # if '(' in i or ')' in i or '{' in i or '}' in i or ';' in i or ',' in i:
        #     continue
        if '(' in i or ')' in i or '{' in i or '}' in i:
            continue
        # 数字也给删除
        if '0' in i or '1' in i or '2' in i or '3' in i or '4' in i or '5' in i or '6' in i or '7' in i or '8' in i or '9' in i:
            i = 'number'
        word_list.append(i.strip())

    word_set = list(set(word_list))
    vocab2id = {w: i + 1 for i, w in enumerate(word_set)}
    vocab2id["PAD"] = 0

    # 将某个漏洞数据集下的词汇表保存
    vocab_dir = sc_data_path.split("/")[0]
    vocab_path = os.path.join(vocab_dir,  sc_type + "_vocab_id.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab2id, f)

    print("We have got vocab file! This file contains {0} vocab".format(len(vocab2id)))


# 获取词汇表，用于one-hot向量
def word_process_max_num(sc_data_path, sc_type, max_num):
    word_dict = {}

    all_sol_data = get_all_data(sc_data_path)

    # 清除一些噪声字符
    for i in all_sol_data:
        if i.strip() in ['{', '}', '(', ')', '[', ']', ' ', ';', ',', '//', '.', '**', '!', '/*', '*/']:
            continue
        if '{' in i or '}' in i or ';' in i or '===' in i:
            continue
        # 数字也给删除
        if '0x' in i or '1' in i or '2' in i or '3' in i or '4' in i or '5' in i or '6' in i or '7' in i or '8' in i or '9' in i:
            i = 'number'

        if i.strip() in word_dict:
            word_dict[i.strip()] += 1
        else:
            word_dict[i.strip()] = 1

    if len(word_dict.items()) > max_num:
        a1 = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
        a2 = a1[:max_num]
        word_set = []
        for j in a2:
            word_set.append(j[0])
        vocab2id = {w: i + 1 for i, w in enumerate(word_set)}
    else:
        # word_set = list(set(word_list))
        vocab2id = {w: i + 1 for i, w in enumerate(list(word_dict.keys()))}
    vocab2id["PAD"] = 0
    # print("vocab2id:", vocab2id)
    # 将某个漏洞数据集下的词汇表保存
    vocab_dir = sc_data_path.split("/")[0]
    vocab_path = os.path.join(vocab_dir,  sc_type + "_vocab_id.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab2id, f)

    print("We have got vocab file! This file contains {0} vocab".format(len(vocab2id)))


if __name__ == "__main__":
    # RE(2548_0_2123_1_425)-vocab 107877; TD(4124_0_3435_1_689)-vocab 150621
    # URV(1165_0_971_1_194)-vocab 59531; BN(1802_0_1059_1_743)-vocab 26519
    # simlify
    # RE(2548_0_2123_1_425)-vocab 30225; TD(4124_0_3435_1_689)-vocab 42065
    # URV(1165_0_971_1_194)-vocab 25696; BN(1802_0_1059_1_743)-vocab 11121
    # AC(906_0_755_1_151)-vocab 19823
    # new
    # RE(2548_0_2123_1_425)-vocab 23956; TD(4124_0_3435_1_689)-vocab 27980
    # URV(1165_0_971_1_194)-vocab 13880; BN(1802_0_1059_1_743)-vocab 11123
    # AC(906_0_755_1_151)-vocab 11219
    sc_data_path = r"SU/all_sol.json"
    sc_type = "SU"

    # word_process(sc_data_path, sc_type)
    word_process_max_num(sc_data_path, sc_type, max_num=9990)