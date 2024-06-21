"""
VDNLL算法：Noisy Label Learning for Smart Contract Vulnerability Detection
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from Arch_SE.vdnll_tool import select_clean_samples
from copy import deepcopy
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, args, model_1, model_2, log_set, device):
        # 设置model
        self.device, self.batch_size = device, args.batch_size
        self.model_1, self.model_2 = model_1.to(self.device), model_2.to(self.device)

        self.lr, self.nb_warmup = args.learning_rate, 10

        self.log_set = log_set
        self.log_set.info('--------------------VDNLL-------------------')

        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(list(self.model_1.parameters()) + list(self.model_2.parameters()), lr=self.lr)


    def extract_features(self, model, train_x):
        # extract deep features of training examples
        model.eval()

        feature, logits = [], []
        idx = int(len(train_x) / self.batch_size)
        with torch.no_grad():
            for i in range(idx+1):
                input = train_x[i*self.batch_size: min((i+1)*self.batch_size, len(train_x))]
                input = torch.LongTensor(input).to(self.device)

                feat, _, logit = model(input)
                if torch.cuda.is_available():
                    feature.extend(feat.data.cpu().tolist())
                    logits.extend(logit.data.cpu().tolist())
                else:
                    feature.extend(feat.data.tolist())
                    logits.extend(logit.data.tolist())
        return np.array(feature), np.array(logits)

    def train_iteration(self, data_loader):
        label_acc, all_loss = 0., 0.
        num_step = 0

        for inputs, targets in data_loader:
            num_step += 1
            lbs = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            _, _, output_1 = self.model_1(inputs)
            loss_1 = self.CE(output_1, targets)

            _, _, output_2 = self.model_2(inputs)
            loss_2 = self.CE(output_2, targets)

            self.optimizer.zero_grad()
            loss_1.backward()
            loss_2.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = targets.eq(output_1.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            all_loss += loss_1.item() + loss_2.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))


    def run_train(self, train_data, data_loader, ep):
        # the standard training: only the CE loss is use

        X_train, Y_train = train_data

        with torch.enable_grad():
            if ep < self.nb_warmup:
                self.log_set.info("Warm-up Training")
                self.model_1.train()
                self.model_2.train()

                self.train_iteration(data_loader)
            else:
                self.log_set.info("Training with noisy labels")
                # select the clean samples
                features_1, _ = self.extract_features(self.model_1, X_train)
                _, logit_2 = self.extract_features(self.model_2, X_train)
                clean_x, clean_y = select_clean_samples(features_1, logit_2, X_train, Y_train)

                self.log_set.info(">>>>>[train] the shape of clean data is {0}".format(clean_x.shape))

                clean_x, clean_y = torch.LongTensor(clean_x), torch.LongTensor(clean_y)
                # build new data loader using the selected clean samples
                dataset = Data.TensorDataset(clean_x, clean_y)
                data_loader_train = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                self.model_1.train()
                self.model_2.train()
                self.train_iteration(data_loader_train)

    def val_iteration(self, data_loader):
        acc_1, acc_2, num_step = 0., 0., 0
        for data, targets in data_loader:
            num_step += 1
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, _, logits1 = self.model_1(data)
            _, _, logits2 = self.model_2(data)

            test_acc_1 = targets.eq(logits1.max(1)[1]).float().sum().item()
            acc_1 += test_acc_1 / data.size(0)

            test_acc_2 = targets.eq(logits2.max(1)[1]).float().sum().item()
            acc_2 += test_acc_2 / data.size(0)

        acc_1, acc_2 = acc_1 / float(num_step), acc_2 / float(num_step)
        self.log_set.info(
            ">>>>>[test] valid data_1's accuracy is {0}, valid data_2's accuracy is {1}".format(acc_1, acc_2))
        return (acc_1 + acc_2) / 2


    def validate(self, data_loader):
        self.model_1.eval()
        self.model_2.eval()

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

        # print("pred_list shape is {0}, and y_list shape is {1}".format(np.array(pred_list).shape, np.array(y_list).shape))
        tn, fp, fn, tp = confusion_matrix(y_list, np.argmax(pred_list, axis=1)).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)

        recall, precision = tp / (tp + fn + 0.000001), tp / (tp + fp + 0.000001)
        F1 = (2 * precision * recall) / (precision + recall + 0.000001)

        return acc, recall, precision, F1

    # 主函数
    def loop(self, epochs, train_data, train_loader, val_data, test_data):

        best_acc, best_epoch = 0., 0
        for ep in range(epochs):
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))

            self.run_train(train_data, train_loader, ep)
            val_acc = self.validate(val_data)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = ep
                self.best_model = deepcopy(self.model_1).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model_1, test_data)
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


if __name__ == '__main__':
    pass