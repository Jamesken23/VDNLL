import torch, os, pickle
import torch.utils.data as Data
from imblearn.over_sampling import SMOTE

from Arch_CV import PL, Co_Teaching, Co_Teaching_plus, Decoupling, JoCoR, Co_Learning

from Utils.log_helper import get_logger, get_log_path
from Utils.data_utils import DataSetWarpper
from Datasets.load_data import load_train_valid_test_data, get_training_data_with_noisy_labels, get_data_path_vocab
from Utils.config import create_parser
from Networks import lstm, transformer
from Arch_CV.Colearning_Loader import create_all_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = create_parser()


def get_balanced_data(inputs, labels):

    sm = SMOTE(random_state=39)
    inputs, labels = sm.fit_resample(inputs, labels.ravel())

    return inputs, labels


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
    train_data_clean, train_data_noise, train_label_clean, train_label_noise = get_training_data_with_noisy_labels(train_data,
                                                             train_label, is_symmetric, mislabel_rate, num_classes)
    log_set.info("Nosiy data shape information: Noisy train data is {0}, clean train data is {1}".format(
        train_data_noise.shape, train_data_clean.shape))

    if args.is_balanced:
        train_data_clean, train_label_clean = get_balanced_data(train_data_clean, train_label_clean)
        train_data_noise, train_label_noise = get_balanced_data(train_data_noise, train_label_noise)
        valid_data, valid_label = get_balanced_data(valid_data, valid_label)
        test_data, test_label = get_balanced_data(test_data, test_label)

    log_set.info("Balanced data shape information: Noisy train data is {0}, clean train data is {1}".format(
        train_data_noise.shape, train_data_clean.shape))


    train_c_inputs, train_c_labels = torch.LongTensor(train_data_clean), torch.LongTensor(train_label_clean)
    train_n_inputs, train_n_labels = torch.LongTensor(train_data_noise), torch.LongTensor(train_label_noise)
    valid_inputs, valid_labels = torch.LongTensor(valid_data), torch.LongTensor(valid_label)
    test_inputs, test_labels = torch.LongTensor(test_data), torch.LongTensor(test_label)
    
    # 加载训练数据集
    train_c_dataset = Data.TensorDataset(train_c_inputs, train_c_labels)
    train_n_dataset = Data.TensorDataset(train_n_inputs, train_n_labels)
    # 加载验证集
    valid_dataset = Data.TensorDataset(valid_inputs, valid_labels)
    # 加载测试数据集
    test_dataset = Data.TensorDataset(test_inputs, test_labels)
    
    if args.data_idx:
        train_c_dataset = DataSetWarpper(train_c_dataset, num_classes)
        train_n_dataset = DataSetWarpper(train_n_dataset, num_classes)

    train_c_loader = Data.DataLoader(train_c_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    n_batch_size = int(args.batch_size*len(train_data_noise)/len(train_data_clean))
    train_n_loader = Data.DataLoader(train_n_dataset, batch_size=n_batch_size, shuffle=True, num_workers=2)
    
    val_loader = Data.DataLoader(valid_dataset, batch_size=len(valid_data), num_workers=2)
    test_loader = Data.DataLoader(test_dataset, batch_size=len(test_data), num_workers=2)
    
    return train_c_loader, train_n_loader, val_loader, test_loader


if __name__ == '__main__':

    args.model, args.arch, args.SC_Type = "LSTM", "Co_Learning", "RE"
    args.data_idx, args.weak_twice = False, False
    args.is_symmetric = True
    # args.is_balanced = False

    # 开始打印日志信息
    log_path, log_name = get_log_path(args)
    log_set = get_logger(log_path)

    if args.arch == "PL":
        args.data_idx, args.is_symmetric = True, False
        # 获取embedding model
        emb_model = create_model(args)

        train_c_loader, train_n_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = PL.Trainer(emb_model, log_set, device)

        sc_train.loop(args.epochs, train_c_loader, train_n_loader, val_loader, test_loader)

    elif args.arch == "Co_Teaching":
        args.data_idx = True

        emb_model_1, emb_model_2 = create_model(args), create_model(args)
        train_c_loader, train_n_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = Co_Teaching.Trainer(args, emb_model_1, emb_model_2, log_set, device)

        sc_train.loop(args.epochs, train_c_loader, train_n_loader, val_loader, test_loader)

    elif args.arch == "Co_Teaching_plus":
        args.data_idx = True

        emb_model_1, emb_model_2 = create_model(args), create_model(args)
        train_c_loader, train_n_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = Co_Teaching_plus.Trainer(args, emb_model_1, emb_model_2, log_set, device)

        sc_train.loop(args.epochs, train_c_loader, train_n_loader, val_loader, test_loader)

    elif args.arch == "Decoupling":
        args.data_idx = True

        emb_model_1, emb_model_2 = create_model(args), create_model(args)
        train_c_loader, train_n_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = Decoupling.Trainer(args, emb_model_1, emb_model_2, log_set, device)

        sc_train.loop(args.epochs, train_c_loader, train_n_loader, val_loader, test_loader)

    elif args.arch == "JoCoR":
        args.data_idx = True

        emb_model_1, emb_model_2 = create_model(args), create_model(args)
        train_c_loader, train_n_loader, val_loader, test_loader = create_loader(args, log_set, args.is_symmetric)
        sc_train = JoCoR.Trainer(args, emb_model_1, emb_model_2, log_set, device)

        sc_train.loop(args.epochs, train_c_loader, train_n_loader, val_loader, test_loader)

    elif args.arch == "Co_Learning":
        args.weak_twice = True

        emb_model_1, emb_model_2 = create_model(args), create_model(args)
        train_loader, val_loader, test_loader = create_all_loader(args, log_set, args.is_symmetric)
        sc_train = Co_Learning.Trainer(args, emb_model_1, log_set, device)

        sc_train.loop(args.epochs, train_loader, val_loader, test_loader)