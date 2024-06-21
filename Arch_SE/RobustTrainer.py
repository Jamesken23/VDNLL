"""
RobustTrainer算法
Ref: https://github.com/RobustTrainer/RobustTrainer/blob/main/feature_learning.py
"""

import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from Arch_SE.RT_tool import get_prototypes, select_clean_samples
from copy import deepcopy
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, args, model, log_set, device):
        # 设置model
        self.device, self.batch_size, self.nb_classes = device, args.batch_size, args.num_classes
        self.model = model.to(self.device)
        self.lr, self.nb_warmup, self.temperature = args.learning_rate, 5, 0.5
        self.w_ce, self.w_cl = 1.0, 1.0

        self.log_set = log_set
        self.log_set.info('--------------------RobustTrainer-------------------')

        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.NLL = nn.NLLLoss().to(self.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)

    def train_warmup(self, data_loader):
        label_acc, all_loss = 0., 0.
        num_step = 0

        for inputs, targets in data_loader:
            num_step += 1
            lbs = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            _, _, outputs = self.model(inputs)
            loss = self.CE(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            all_loss += loss.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))

    def train_iteration(self, data_loader, nb_prototypes):
        # train the model with both contrastive and CE loss
        label_acc, all_loss = 0., 0.
        num_step = 0

        for inputs, targets in data_loader:
            num_step += 1
            lbs = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            features, intermediate_feat, outputs = self.model(inputs)
            # obtain the class probabilities of prototypes
            class_prototypes = copy.deepcopy(self.prototypes)
            class_prototypes = torch.from_numpy(class_prototypes).float().to(self.device)
            # 对应score(x, y)
            logits_proto = torch.mm(features, class_prototypes.t()) / self.temperature
            softmax_proto = F.softmax(logits_proto, dim=1)
            prob_proto = torch.zeros((softmax_proto.shape[0], self.nb_classes), dtype=torch.float32).to(self.device)
            for i in range(self.nb_classes):
                prob_proto[:, i] = torch.sum(
                    softmax_proto[:, i * nb_prototypes: (i + 1) * nb_prototypes], dim=1)
            # contrastive loss
            cl_loss = self.NLL(torch.log(prob_proto + 1e-5), targets)
            # classification loss
            ce_loss = self.CE(outputs, targets)
            loss = self.w_ce * ce_loss + self.w_cl * cl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs
            all_loss += cl_loss.item() + ce_loss.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))

    def run_train(self, data_loader, nb_prototypes, is_warmup=False):
        # the standard training: only the CE loss is used
        self.model.train()

        with torch.enable_grad():
            if is_warmup:
                self.log_set.info("Warm-up Training")
                self.train_warmup(data_loader)
            else:
                self.log_set.info("Training with noisy labels")
                self.train_iteration(data_loader, nb_prototypes)

    def extract_features(self, train_x):
        # extract deep features of training examples
        self.model.eval()

        feature = []
        idx = int(len(train_x) / self.batch_size)
        with torch.no_grad():
            for i in range(idx+1):
                input = train_x[i*self.batch_size: min((i+1)*self.batch_size, len(train_x))]
                input = torch.LongTensor(input).to(self.device)

                feat, _, _ = self.model(input)
                if torch.cuda.is_available():
                    feature.extend(feat.data.cpu().tolist())
                else:
                    feature.extend(feat.data.tolist())
        return np.array(feature)

    def val_iteration(self, data_loader):
        acc_1, acc_2, num_step = 0., 0., 0
        for data, targets in data_loader:
            num_step += 1
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, _, logits1 = self.model(data)

            test_acc_1 = targets.eq(logits1.max(1)[1]).float().sum().item()
            acc_1 += test_acc_1 / data.size(0)

        acc_1 = acc_1 / float(num_step)
        self.log_set.info(
            ">>>>>[test] valid data's accuracy is {0}".format(acc_1))
        return acc_1

    def validate(self, data_loader):
        self.model.eval()

        with torch.no_grad():
            return self.val_iteration(data_loader)

    def predict(self, model, data_loader):
        model.eval()

        pred_list, y_list = [], []
        for data, targets in data_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, _, logits = model(data)

            if torch.cuda.is_available():
                y_label = targets.cpu().detach().numpy().tolist()
                pred = logits.cpu().detach().numpy().tolist()
            else:
                y_label = targets.detach().numpy().tolist()
                pred = logits.detach().numpy().tolist()

            pred_list.extend(pred)
            y_list.extend(y_label)

        tn, fp, fn, tp = confusion_matrix(y_list, np.argmax(pred_list, axis=1)).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)

        recall, precision = tp / (tp + fn + 0.000001), tp / (tp + fp + 0.000001)
        F1 = (2 * precision * recall) / (precision + recall + 0.000001)

        return acc, recall, precision, F1

    # 主函数
    def loop(self, epochs, train_data, train_loader, val_data, test_data):
        X_train, Y_train = train_data
        nb_prototypes = int((len(X_train)/2) ** 0.5)

        best_acc, best_epoch = 0., 0
        for ep in range(epochs):
            self.ep = ep
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))

            if ep < self.nb_warmup:
                # warmup
                self.run_train(train_loader, nb_prototypes, is_warmup=True)
            else:
                """
                Each epoch contains the following two steps:
                1. Detecting mislabeled samples;
                2. Updating feature extractor
                """
                # extract deep features of training examples
                features = self.extract_features(X_train)

                # select the clean samples
                clean_feat, clean_y, clean_idx = select_clean_samples(features, Y_train, nb_prototypes)
                clean_x = [X_train[i] for i in clean_idx]

                # refine the class prototypes using the newly selected clean samples
                class_prototypes = get_prototypes(clean_feat, clean_y, nb_prototypes)
                class_prototypes = np.vstack(class_prototypes)
                self.prototypes = class_prototypes

                weak_x, weak_y = torch.LongTensor(clean_x), torch.LongTensor(clean_y)
                # build new data loader using the selected clean samples
                dataset = Data.TensorDataset(weak_x, weak_y)
                data_loader_train = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                self.run_train(data_loader_train, nb_prototypes, is_warmup=False)

            val_acc = self.validate(val_data)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = ep
                self.best_model = deepcopy(self.model).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model, test_data)
        self.log_set.info(
            "Final epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(epochs,
                                                                                                            acc,
                                                                                                            recall,
                                                                                                            precision,
                                                                                                            F1))
        acc, recall, precision, F1 = self.predict(self.best_model, test_data)
        self.log_set.info(
            "The best epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(
                best_epoch, acc, recall, precision, F1))