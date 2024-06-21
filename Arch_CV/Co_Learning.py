"""
Co-Learning算法
Ref: https://github.com/chengtan9907/Co-learning/blob/master/algorithms/Colearning.py
"""

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

import numpy as np
from Arch_CV.Teaching_Loss import loss_coteaching, loss_structrue
from Arch_CV.NTXentLoss import NTXentLoss
from copy import deepcopy
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, args, model, log_set, device):

        # 设置model
        self.device, self.batch_size = device, args.batch_size

        self.model_scratch = model.to(self.device)

        self.lr, self.epochs = args.learning_rate, args.epochs
        epoch_decay_start, num_gradual, exponent = 20, 1, 1
        self.adjust_lr = 1.0

        self.log_set = log_set
        self.log_set.info('--------------------Co-Learning-------------------')

        # Adjust learning rate and betas for Adam Optimizer
        mom1, mom2 = 0.9, 0.1
        self.alpha_plan = [self.lr] * self.epochs
        self.beta1_plan = [mom1] * self.epochs

        for i in range(epoch_decay_start, self.epochs):
            self.alpha_plan[i] = float(self.epochs - i) / (self.epochs - epoch_decay_start) * self.lr
            self.beta1_plan[i] = mom2

        self.optimizer1 = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(list(self.model_scratch.fc.parameters()), lr=self.lr / 5)



    def mixup_data(self, x, y, alpha=5.0):
        lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample() if alpha > 0 else 1
        index = torch.randperm(x.size()[0]).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_x = torch.tensor(mixed_x, dtype=torch.long)
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam=0.5):
        return (lam * F.cross_entropy(pred, y_a, reduce=False) + (1 - lam) * F.cross_entropy(pred, y_b,
                                                                                       reduce=False)).mean()

    def train_iteration(self, train_loader):
        label_acc, all_loss = 0., 0.
        num_step = 0
        for (data_raw, data_pos_1, data_pos_2, label_y) in train_loader:
            num_step += 1

            # === decode targets of unlabeled data ===
            lbs = data_raw.size(0)
            # print("clean_x shape is {0}, noisy_x shape is {1}, c_n_x shape is {2}".format(clean_x.shape, noisy_x.shape, c_n_x.shape))
            c_n_raw, c_n_pos_1, c_n_pos_2, c_n_y = data_raw.to(self.device), data_pos_1.to(self.device), data_pos_2.to(self.device), label_y.to(self.device)

            # === forward ===
            feat, outs, logits = self.model_scratch(c_n_raw)
            loss_feat = loss_structrue(outs.detach(), logits)

            # backward
            self.optimizer2.zero_grad()
            loss_feat.backward()
            self.optimizer2.step()

            # Self-learning, out_1和out_2指代projection
            out_1 = self.model_scratch(c_n_pos_1, ignore_feat=True, forward_fc=False)
            out_2 = self.model_scratch(c_n_pos_2, ignore_feat=True, forward_fc=False)

            ntxent = NTXentLoss(self.device, lbs, temperature=0.5, use_cosine_similarity=True)
            loss_con = ntxent(out_1, out_2)

            # feat, outs, logits = self.model_scratch(c_n_raw)

            # Supervised-learning
            inputs, targets_a, targets_b, lam = self.mixup_data(c_n_raw, c_n_y, alpha=5.0)
            _, logits = self.model_scratch(inputs, ignore_feat=True)
            loss_sup = self.mixup_criterion(logits, targets_a, targets_b, lam)

            # Loss
            loss = loss_sup + loss_con

            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

            ##=== log info ===
            temp_acc = c_n_y.eq(logits.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            all_loss += loss.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))

    def train(self, train_loader):
        print('Start training ...')
        self.model_scratch.train()

        if self.adjust_lr:
            self.adjust_learning_rate(self.optimizer1, self.ep)
            self.adjust_learning_rate(self.optimizer2, self.ep)

        with torch.enable_grad():
            self.train_iteration(train_loader)

    def val_iteration(self, data_loader):
        acc_1, acc_2, num_step = 0., 0., 0
        for data, targets in data_loader:
            num_step += 1
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, _, logits1 = self.model_scratch(data)

            test_acc_1 = targets.eq(logits1.max(1)[1]).float().sum().item()
            acc_1 += test_acc_1 / data.size(0)

        acc_1 = acc_1 / float(num_step)
        self.log_set.info(
            ">>>>>[test] valid data's accuracy is {0}".format(acc_1))
        return acc_1

    def validate(self, data_loader):
        self.model_scratch.eval()

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
    def loop(self, epochs, train_c_data, val_data, test_data):

        best_acc, best_epoch = 0., 0
        for ep in range(epochs):
            self.ep = ep
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))
            self.train(train_c_data)

            val_acc = self.validate(val_data)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = ep
                self.best_model = deepcopy(self.model_scratch).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model_scratch, test_data)
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