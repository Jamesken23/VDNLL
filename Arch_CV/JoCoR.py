# -*- coding:utf-8 -*-
"""
JoCR算法
Ref: https://github.com/chengtan9907/Co-learning/blob/master/algorithms/JoCoR.py
"""

import torch
import torch.nn.functional as F

import numpy as np
from Arch_CV.Teaching_Loss import loss_jocor
from copy import deepcopy
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, args, model1, model2, log_set, device):

        # 设置model
        self.device, self.co_lambda = device, 0.1
        self.model1, self.model2 = model1.to(self.device), model2.to(self.device)

        self.lr, self.epochs = args.learning_rate, args.epochs

        forget_rate = args.forget_rate
        epoch_decay_start, num_gradual, exponent = 20, 1, 1
        self.adjust_lr = 1.0

        self.log_set  = log_set
        self.log_set.info('--------------------JoCoR -------------------')

        # Adjust learning rate and betas for Adam Optimizer
        mom1, mom2 = 0.9, 0.1
        self.alpha_plan = [self.lr] * self.epochs
        self.beta1_plan = [mom1] * self.epochs

        for i in range(epoch_decay_start, self.epochs):
            self.alpha_plan[i] = float(self.epochs - i) / (self.epochs - epoch_decay_start) * self.lr
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(self.epochs) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

        self.optimizer = torch.optim.AdamW(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr)

        self.loss_fn = loss_jocor

    def train_iteration(self, train_c_loader, train_n_loader):
        label_acc, all_loss = 0., 0.
        num_step = 0
        for (clean_x, clean_y, _), (noisy_x, noisy_y, _) in zip(train_c_loader, train_n_loader):
            num_step += 1
            # 对clean_x和noisy_x进行拼接
            c_n_x = torch.cat((clean_x, noisy_x), 0)
            # 对clean_y和noisy_y进行拼接
            c_n_y = torch.cat((clean_y, noisy_y), 0)
            # === decode targets of unlabeled data ===
            lbs = c_n_x.size(0)
            # print("clean_x shape is {0}, noisy_x shape is {1}, c_n_x shape is {2}".format(clean_x.shape, noisy_x.shape, c_n_x.shape))
            c_n_x, c_n_y = c_n_x.to(self.device), c_n_y.to(self.device)

            # === forward ===
            _, _, logits1 = self.model1(c_n_x)
            _, _, logits2 = self.model2(c_n_x)

            loss_1, loss_2 = self.loss_fn(logits1, logits2, c_n_y, self.rate_schedule[self.ep], self.co_lambda)

            ## backward
            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = c_n_y.eq(logits1.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            all_loss += loss_1.item() + loss_2.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))

    def train(self, train_c_loader, train_n_loader):
        print('Start training ...')
        self.model1.train()
        self.model2.train()

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, self.ep)

        with torch.enable_grad():
            self.train_iteration(train_c_loader, train_n_loader)

    def val_iteration(self, data_loader):
        acc_1, acc_2, num_step = 0., 0., 0
        for data, targets in data_loader:
            num_step += 1
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, _, logits1 = self.model1(data)
            _, _, logits2 = self.model2(data)

            test_acc_1 = targets.eq(logits1.max(1)[1]).float().sum().item()
            acc_1 += test_acc_1 / data.size(0)

            test_acc_2 = targets.eq(logits2.max(1)[1]).float().sum().item()
            acc_2 += test_acc_2 / data.size(0)

        acc_1, acc_2 = acc_1 / float(num_step), acc_2 / float(num_step)
        self.log_set.info(
            ">>>>>[test] valid data_1's accuracy is {0}, valid data_2's accuracy is {1}".format(acc_1, acc_2))
        return (acc_1 + acc_2) / 2

    def validate(self, data_loader):
        self.model1.eval()
        self.model2.eval()

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
    def loop(self, epochs, train_c_data, train_n_data, val_data, test_data):

        best_acc, best_epoch = 0., 0
        for ep in range(epochs):
            self.ep = ep
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))
            self.train(train_c_data, train_n_data, )

            val_acc = self.validate(val_data)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = ep
                self.best_model = deepcopy(self.model1).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model1, test_data)
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

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1


if __name__ == '__main__':
        pass