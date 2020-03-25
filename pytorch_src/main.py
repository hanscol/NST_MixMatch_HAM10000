from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from pytorch_src.data import *
from pytorch_src.utils import *
from pytorch_src.model import *
from pytorch_src.batch import *
import random
import argparse
import numpy as np
import os


def create_model(device, ema=False):
    model = models.densenet169(pretrained=True)
    ft = model.classifier.in_features
    model.classifier = torch.nn.Linear(ft, 7)
    model = model.to(device)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999, lr=0.0001):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def test_model(config):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_dataset = Test_Dataset(config.test_file, config)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    model = create_model(device, ema=False)

    model_file = '{}/model'.format(config.out_dir)
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    loss, accuracy, conf_matrix = test(model, device, test_loader)

    acc_file = '{}/test_accuracy.txt'.format(config.out_dir)
    conf_file = '{}/confusion'.format(config.out_dir)

    # with open(acc_file, 'w') as f:
    #     f.write(str(accuracy))
    #
    # np.save(conf_file, np.array(conf_matrix))

    return accuracy

def train_model(config):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = Train_Dataset(config.labeled_train_file, config.unlabeled_train_file, config)

    val_dataset = Test_Dataset(config.val_file, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    model = create_model(device)
    ema_model = create_model(device, ema=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    ema_optimizer= WeightEMA(model, ema_model, alpha=config.ema_decay, lr=config.lr)

    train_loss_file = '{}/train_loss.txt'.format(config.out_dir)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = '{}/validate_loss.txt'.format(config.out_dir)
    f = open(validate_loss_file, 'w')
    f.close()
    train_accuracy_file = '{}/train_accuracy.txt'.format(config.out_dir)
    f = open(train_accuracy_file, 'w')
    f.close()
    validate_accuracy_file = '{}/validate_accuracy.txt'.format(config.out_dir)
    f = open(validate_accuracy_file, 'w')
    f.close()

    model_file = '{}/model'.format(config.out_dir)


    if config.continue_training:
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)

    seq_increase = 0
    min_loss = 10000

    for epoch in range(1, config.total_epochs+1):
        loss = train(model, device, train_loader, optimizer, config, ema_optimizer, epoch)

        with open(train_loss_file, "a") as file:
            file.write(str(loss))
            file.write('\n')

        loss, accuracy, confusion = test(ema_model, device, val_loader)


        class_accuracies = '\t\tAccuracy by class: '
        for i in range(len(confusion)):
            correct = confusion[i,i]
            total = sum(confusion[:,i])
            class_accuracies += '{0}--{1:.2f}%   '.format(i, (correct/total)*100)

        print(class_accuracies)

        if epoch % 5  ==0 and (config.mixmatch or config.null_space_tuning) and epoch != config.total_epochs:
            train_dataset.assign_class(model, device)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)


        with open(validate_loss_file, "a") as file:
            file.write(str(loss))
            file.write('\n')
        with open(validate_accuracy_file, "a") as file:
            file.write(str(accuracy))
            file.write('\n')

        if loss < min_loss:
            min_loss = loss
            with open(model_file, 'wb') as f:
                torch.save(ema_model.state_dict(), f)

        elif config.early_stop != -1:
            if loss > min_loss:
                seq_increase += 1
                if seq_increase == config.early_stop:
                    break
            else:
                seq_increase = 0


        if config.decay_epoch != -1:
            if epoch % config.decay_epoch == 0:
                config.lr = config.lr * config.decay_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.lr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--null_space_tuning', action='store_true',
                        help='Turn on null space training')
    parser.add_argument('--alpha', type=float, default=10,
                        help='Determines the impact of null space tuning')
    parser.add_argument('--mixup_alpha', type=float, default=0.75,
                        help='Determines contribution of paired image in MixUp')
    parser.add_argument('--lambda_u', type=float, default=25.0,
                        help='Determines contribution of MixMatch Loss')
    parser.add_argument('--T', type=float, default=0.5,
                        help='Determines sharpening temperature for entropy minimization while guessing labels for unlabeled data')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='Decay for exponential moving average')
    parser.add_argument('--mixmatch', action='store_true',
                        help='Turn on MixMatch')
    parser.add_argument('--labeled_train_file', type=str, default='train_labeled.txt',
                        help='This file should contain the path to an image as well and an integer specifying its class (comma separated) on each line')
    parser.add_argument('--unlabeled_train_file', type=str, default='train_unlabeled.txt',
                        help='This file should contain the path to an image as well and an integer specifying its class (comma separated) on each line')
    parser.add_argument('--val_file', type=str, default='val.txt',
                        help='This file should contain the path to an image as well and an integer specifying its class (comma separated) on each line')
    parser.add_argument('--test_file', type=str, default='test.txt',
                        help='This file should contain the path to an image as well and an integer specifying its class (comma separated) on each line')
    parser.add_argument('--out_dir', type=str, default='out/',
                        help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train',
                        help='Determines whether to backpropagate or not')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Decides size of each training batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Optimizer\'s learning rate')
    parser.add_argument('--total_epochs', type=int, default=75,
                        help='Maximum number of epochs for training')
    parser.add_argument('--early_stop', type=int, default=-1,
                        help='Decide to stop early after this many epochs in which the validation loss increases (-1 means no early stopping)')
    parser.add_argument('--decay_epoch', type=int, default=-1,
                        help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--decay_rate', type=float, default=0.10,
                        help='Rate at which the learning rate will be decayed')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from saved model')
    parser.add_argument('--rampup', default=75, type=int, metavar='N',
                        help='number of epochs before unlabeled terms reach full value')

    config = parser.parse_args()

    print(config)

    os.makedirs(config.out_dir, exist_ok=True)

    if config.mode == 'train':
        train_model(config)
    elif config.mode == 'test':
        test_model(config)

if __name__ == '__main__':
    main()
