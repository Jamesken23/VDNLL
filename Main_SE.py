import torch, os, pickle
import torch.utils.data as Data
from imblearn.over_sampling import SMOTE
import numpy as np

from Arch_SE import RobustTrainer, PILOT, VDNLL

from Utils.log_helper import get_logger, get_log_path
from Utils.data_utils import DataSetWarpper, TransformWeakTwice
from Datasets.load_data import load_train_valid_test_data, get_training_data_with_noisy_labels, get_data_path_vocab
from Utils.config import create_parser
from Networks import lstm, transformer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = create_parser()


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

# 获取base model，这个是embedding model
def create_model(args):
    model_name, vocab_size, embedding_dim, num_classes = args.model, args.vocab_size, args.embedding_dim, args.num_classes
    max_setence_length = args.max_setence_length

    if model_name == "LSTM":
        model = lstm.LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, num_classes=num_classes, bid=False)
    elif model_name == "Transformer":
        model = transformer.Transformer(vocab_size, max_setence_length, device)
    else:
        print("We do not set this model!")
        exit()
    # print("Your model architecture is {0}".format(model))
    return model


def create_loader(args, log_set, is_symmetric=False):
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

    all_data_label = (train_data, train_label)

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

    return all_data_label, train_loader, val_loader, test_loader


if __name__ == '__main__':

    args.model, args.arch, args.SC_Type = "LSTM", "VDNLL", "RE"
    args.data_idx, args.weak_twice = False, False
    args.is_symmetric = True
    # args.is_balanced = False

    # 开始打印日志信息
    log_path, log_name = get_log_path(args)
    log_set = get_logger(log_path)

    if args.arch == "RobustTrainer":
        # 获取embedding model
        emb_model = create_model(args)

        all_data_label, train_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = RobustTrainer.Trainer(args, emb_model, log_set, device)

        sc_train.loop(args.epochs, all_data_label, train_loader, val_loader, test_loader)
    elif args.arch == "PILOT":
        # 获取embedding model
        emb_model = create_model(args)

        all_data_label, train_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = PILOT.Trainer(args, emb_model, log_set, device)

        sc_train.loop(args.epochs, all_data_label, train_loader, val_loader, test_loader)
    elif args.arch == "VDNLL":
        # 获取embedding model
        emb_model_1, emb_model_2 = create_model(args), create_model(args)

        all_data_label, train_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = VDNLL.Trainer(args, emb_model_1, emb_model_2, log_set, device)

        sc_train.loop(args.epochs, all_data_label, train_loader, val_loader, test_loader)