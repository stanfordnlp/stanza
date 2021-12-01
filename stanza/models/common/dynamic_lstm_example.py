"""
Demo code for the DynamicLSTM

Runs a number sequence prediction task

Adapted from https://github.com/v1xerunt/PyTorch-Dynamic_LSTM

MIT License

Copyright (c) 2019 v1xerunt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

from stanza.models.common.dynamic_lstm import DynamicLSTM

def train_model(train_dataset, test_dataset):
    model = DynamicLSTM(n_actions=10, n_units=10, n_input=10,
                        n_hidden=30, n_output=10, lamda=1, dropout=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

    best_acc = 0
    train_loss = []
    count = 0

    for epoch in range(200):
        print('****************  RL is beginning  *******************')
        model.train()
        cur_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            # data is of shape batch_size x item_len x item_dim
            # so 100 x 10 x 10
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            # output will be, for each item in the batch,just the last item
            # so 100 x 10
            output, (acts, act_prob, cur_h, cur_c, observed_h, observed_c) = model(data)
            loss_task = torch.mean(-torch.log(output+1e-7)*target)
            correct_pred = torch.argmax(output, 1).eq(torch.argmax(target, 1))
            accuracy = torch.mean(correct_pred.float())

            act_prob = torch.stack(act_prob).permute(1, 0)
            acts = torch.squeeze(torch.stack(acts)).permute(1, 0)
            #neg_log_prob = torch.sum(-torch.log(act_prob+0.000001) * onehot[acts], dim=2)
            rewards = (correct_pred.float() - 0.5) * 2
            rewards = rewards.unsqueeze(-1)
            loss_RL = torch.mean(-act_prob * rewards)
            loss_total = loss_task+loss_RL*0.3
            cur_loss.append(loss_total.detach().numpy())
            loss_total.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print('the %d epoch the %d time accuracy is %f, loss is %f' %
                      (epoch, batch_idx, accuracy, loss_total))
                print('loss_RL:', loss_RL.detach().numpy())
                print('loss_Task:', loss_task.detach().numpy())
                print(acts[:5,-1])
                print(torch.argmax(target[:5], dim=1))
                print()
        train_loss.append(np.average(np.array(cur_loss)))

        print('###############  TESTING  ####################')
        model.eval()
        test_loss = []
        test_acc = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data), Variable(target)
            output, (acts, act_prob, cur_h, cur_c, observed_h, observed_c) = model(data)
            loss_task = torch.mean(-torch.log(output+1e-7)*target)
            correct_pred = torch.argmax(output, 1).eq(torch.argmax(target, 1))
            accuracy = torch.mean(correct_pred.float())

            acts = torch.squeeze(torch.stack(acts)).permute(1, 0)

            act_prob = torch.stack(act_prob).permute(1, 0)
            rewards = (correct_pred.float() - 0.5) * 2
            rewards = rewards.unsqueeze(-1)
            loss_RL = torch.mean(act_prob * rewards)
            loss_total = loss_task+loss_RL

            test_loss.append(loss_total)
            test_acc.append(accuracy)
        print('the TEST accuracy is %f, loss is %f' %
              (sum(test_acc)/len(test_acc), sum(test_loss)/len(test_loss)))

        cur_acc = sum(test_acc)/len(test_acc)
        if cur_acc > best_acc:
            best_acc = cur_acc
            print('===============================================>>>> SAVE MODEL')
            count = 0
        count += 1
        if count == 5:
            print('--------------------------------------------->>>>  EARLY STOP!!!')


def main():
    # build a dataset
    # the first 10 numbers are inputs, the last one is label in each row
    all_train = []
    all_test = []
    for _ in range(100000):
        a = np.random.choice(10, 9)
        b = np.random.choice(9, 1)
        c = a[b]
        train = np.append(np.append(a, b), c)
        all_train.append(train)

    for _ in range(10000):
        a = np.random.choice(10, 9)
        b = np.random.choice(9, 1)
        c = a[b]
        test = np.append(np.append(a, b), c)
        all_test.append(test)


    onehot = torch.eye(10)
    all_train = onehot[all_train].numpy()
    all_test = onehot[all_test].numpy()

    train_data = all_train[:, :-1, :]
    train_label = all_train[:, -1, :]
    test_data = all_test[:, :-1, :]
    test_label = all_test[:, -1, :]

    train_dataset = Data.TensorDataset(torch.tensor(
        train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.float32))
    test_dataset = Data.TensorDataset(torch.tensor(
        test_data, dtype=torch.float32), torch.tensor(test_label, dtype=torch.float32))

    train_model(train_dataset, test_dataset)

if __name__ == '__main__':
    main()

