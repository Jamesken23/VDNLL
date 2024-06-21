import torch, os, pickle
import torch.utils.data as Data
from imblearn.over_sampling import SMOTE
import numpy as np

from Utils.data_utils import DataSetWarpper, TransformWeakTwice
from Datasets.load_data import load_train_valid_test_data, get_training_data_with_noisy_labels, get_data_path_vocab



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_balanced_data(inputs, labels):

    sm = SMOTE(random_state=39)
    inputs, labels = sm.fit_resample(inputs, labels.ravel())

    return inputs, labels


def get_vocab_id(sc_type):
    _, _, _, vocab2id_path = get_data_path_vocab(sc_type)
    id2vocab = {}
    # 加载词汇表
    with open(vocab2id_path, 'rb') as f:
        vocab2id = pickle.load(f)

    return vocab2id


def create_all_loader(args, log_set, is_symmetric=False):

    SC_Type, max_setence_length = args.SC_Type, args.max_setence_length

    # 读取训练集、验证集、测试集
    train_data, train_label, valid_data, valid_label, test_data, test_label = load_train_valid_test_data(SC_Type,
                                                                                                         max_setence_length)

    log_set.info("Original shape information: labeled train data is {0}, valid data is {1}, test data is {2}".format(
        train_data.shape, valid_data.shape, test_data.shape))

    # 对数据集进行标签加噪
    mislabel_rate, num_classes = args.mislabel_rate, args.num_classes

    train_data_clean, train_data_noise, train_label_clean, train_label_noise = get_training_data_with_noisy_labels \
        (train_data, train_label, is_symmetric, mislabel_rate, num_classes)
    log_set.info("Nosiy data shape information: Noisy train data is {0}, clean train data is {1}".format(
        train_data_noise.shape, train_data_clean.shape))

    train_data = np.concatenate((train_data_clean, train_data_noise))
    train_label = np.concatenate((train_label_clean, train_label_noise))
    if args.is_balanced:
        train_data, train_label = get_balanced_data(train_data, train_label)
        valid_data, valid_label = get_balanced_data(valid_data, valid_label)
        test_data, test_label = get_balanced_data(test_data, test_label)

    log_set.info("Balanced data shape information: Train data is {0}".format(train_data.shape))


    train_inputs, train_labels = torch.LongTensor(train_data), torch.LongTensor(train_label)
    valid_inputs, valid_labels = torch.LongTensor(valid_data), torch.LongTensor(valid_label)
    test_inputs, test_labels = torch.LongTensor(test_data), torch.LongTensor(test_label)

    # 加载训练数据集
    train_dataset = Data.TensorDataset(train_inputs, train_labels)
    # 加载验证集
    valid_dataset = Data.TensorDataset(valid_inputs, valid_labels)
    # 加载测试数据集
    test_dataset = Data.TensorDataset(test_inputs, test_labels)

    if args.data_idx:
        train_dataset = DataSetWarpper(train_dataset, num_classes)

    if args.weak_twice:
        vocab2id = get_vocab_id(args.SC_Type)
        train_dataset = TransformWeakTwice(train_dataset, num_classes, vocab2id, device)


    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = Data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=2)
    test_loader = Data.DataLoader(test_dataset, batch_size=len(test_data), num_workers=2)

    return train_loader, val_loader, test_loader