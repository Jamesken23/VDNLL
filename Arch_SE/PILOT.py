"""
2023_ASE PILOT算法：When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection
核心要点：每一个epoch中都会选取部分干净的标签（包括vulnerable和non-vulnerable），并且在修复后的数据集中迭代训练
Ref: https://github.com/Eshe0922/PILOT/blob/main/run.py
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from Arch_SE.PI_tool import select_clean_samples
from copy import deepcopy
from sklearn.metrics import confusion_matrix

from pytorch_metric_learning import losses


class Trainer:
    def __init__(self, args, model, log_set, device):
        # 设置model
        self.device, self.batch_size, self.nb_classes = device, args.batch_size, args.num_classes
        self.model = model.to(self.device)
        self.lr, self.nb_warmup = args.learning_rate, 10
        self.con_para = nn.Parameter(torch.ones(1)*0.5).to(self.device)

        self.log_set = log_set
        self.log_set.info('--------------------PILOT-------------------')

        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.ntx_loss = losses.NTXentLoss(temperature=0.1).to(self.device)
        self.con_loss = losses.SupConLoss(temperature=0.1).to(self.device)

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

    def train_iteration(self, data_loader):
        # train the model with both contrastive and CE loss
        label_acc, all_loss = 0., 0.
        num_step = 0

        for inputs, targets in data_loader:
            num_step += 1
            lbs = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            features, intermediate_feat, outputs = self.model(inputs)

            # classification loss
            ce_loss = self.CE(outputs, targets)
            ntx_loss = self.ntx_loss(features, targets.float())
            con_loss = self.con_loss(features, targets.float())

            loss = (1 - self.con_para) * ntx_loss.mean() + self.con_para * con_loss.mean() + ce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs
            all_loss += ntx_loss.mean().item() + con_loss.mean().item() + ce_loss.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step + 0.001), all_loss / float(num_step + 0.001)))

    def run_train(self, data_loader, is_warmup=False):
        # the standard training: only the CE loss is used
        self.model.train()

        with torch.enable_grad():
            if is_warmup:
                self.log_set.info("Warm-up Training")
                self.train_warmup(data_loader)
            else:
                self.log_set.info("Training with noisy labels")
                self.train_iteration(data_loader)

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

        best_acc, best_epoch = 0., 0
        for ep in range(epochs):
            self.ep = ep
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))

            if ep < self.nb_warmup:
                # warmup
                self.run_train(train_loader, is_warmup=True)
            else:
                # extract deep features of training examples
                features = self.extract_features(X_train)

                # select the clean samples
                clean_x, clean_y= select_clean_samples(features, X_train, Y_train)

                self.log_set.info(">>>>>[train] the shape of clean data is {0}".format(clean_x.shape))

                clean_x, clean_y = torch.LongTensor(clean_x), torch.LongTensor(clean_y)
                # build new data loader using the selected clean samples
                dataset = Data.TensorDataset(clean_x, clean_y)
                data_loader_train = Data.DataLoader(dataset, batch_size=self.batch_size)

                self.run_train(data_loader_train, is_warmup=False)

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