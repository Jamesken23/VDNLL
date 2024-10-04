import os, shutil

def copy_sc_data(ori_path, new_path):
    # reentrancy
    # maxcnt_0, max_cnt_1 = 1705, 680

    # Timestamp dependency
    # max_cnt_0, max_cnt_1 = 2248, 2242

    # Integer overflow and underflow
    max_cnt_0, max_cnt_1 = 5815, 1368

    # delegatecall
    max_cnt_0, max_cnt_1 = 278, 136

    cnt_0, cnt_1 = 0, 0
    for i in os.listdir(ori_path):
        if ".sol" not in i:
            continue
        if "_0.sol" in i and cnt_0 < max_cnt_0:
            ori_data_path = os.path.join(ori_path, i)
            new_data_path = os.path.join(new_path, i)
            shutil.copy(ori_data_path, new_data_path)  # 复制文件
            cnt_0 += 1

        if "_1.sol" in i and cnt_1 < max_cnt_1:
            ori_data_path = os.path.join(ori_path, i)
            new_data_path = os.path.join(new_path, i)
            shutil.copy(ori_data_path, new_data_path)  # 复制文件
            cnt_1 += 1

    print("cnt_0 is {0}, and cnt_1 is {1}".format(cnt_0, cnt_1))


def add_sc_data(ori_path, new_path):
    # Timestamp dependency
    # max_cnt_0 = 2248 - 539

    # Integer overflow and underflow
    # max_cnt_0, max_cnt_1 = 5815-2389, 1368-126

    # delegatecall
    max_cnt_0 = 278 - 193

    cnt_0, cnt_1 = 0, 0
    for i in os.listdir(ori_path):
        if ".sol" not in i:
            continue
        if cnt_0 >= max_cnt_0:
            break

        if "_0.sol" in i and cnt_0 < max_cnt_0:
            ori_data_path = os.path.join(ori_path, i)
            new_data_path = os.path.join(new_path, i)
            shutil.copy(ori_data_path, new_data_path)  # 复制文件
            cnt_0 += 1


if __name__ == "__main__":
    ori_path = r"D:\JamesFiles\Todo Tasks\ReadPapers\智能合约数据集\All_team_datasets\Qianpeng_team\DE(371_0_193_1_178)"
    new_path = r"D:\Python\Python_Projects\VDNLL\Datasets\original_sol\DE_414_0_278_1_136"
    copy_sc_data(ori_path, new_path)

    # Timestamp dependency
    ori_path = r"D:\JamesFiles\Todo Tasks\ReadPapers\智能合约数据集\All_team_datasets\Vulhunter_team\Delegatecall(DE_389_0_324_1_65)"
    new_path = r"D:\Python\Python_Projects\VDNLL\Datasets\original_sol\DE_414_0_278_1_136"
    add_sc_data(ori_path, new_path)