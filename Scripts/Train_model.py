#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:27:12 2020

@author: vantaggiato
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

import os
from time import sleep
import datetime

import Utils as utils
#import Augmentation as aug
import Trainer as trainer

torch.manual_seed(0)
workspace = os.path.abspath("../")    # location of checkpoints, scripts and dataset
dataset = 'Dataset3'
data_dir = data_dir = os.path.join(workspace, dataset)
torch.hub.set_dir(workspace)
torch.hub.get_dir()
# ---------------- PARAMETERS ----------------- #
# --------------------------------------------- #
model_name = 'ResNeXt50'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dataparallel = False
gpus = [0,1]
freeze = False
dropout = True
prob = 0.3
lr = 0.000001
momentum = 0.9
step_size = 20
gamma = 0.1
criterion_name = 'Focal'
optimizer_name = 'Adam'
num_classes = 3
num_epochs = 30
batch_size = 64
# --------------------------------------------- #
model, optimizer, criterion, scheduler, transform = utils.create_model(workspace, dataset, num_classes, model_name, freeze, dropout, prob, lr, momentum, step_size, gamma, criterion_name, optimizer_name, device, dataparallel, gpus)
model = model.to(device)

train_dir = os.path.join(data_dir, 'Train')
val_dir = os.path.join(data_dir, 'Val')

# source_dirs = ['Covid-19', 'Normal', 'Pneumonia']
# utils.data_augmentation(workspace, train_dir, source_dirs, False)

train_set = datasets.ImageFolder(train_dir, transform)
val_set = datasets.ImageFolder(val_dir, transform)
train_size = len(train_set)
val_size = len(val_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

model, epochs, train_accs, train_losses, val_accs, val_losses, train_f1_list, val_f1_list = trainer.train(model, train_loader, train_size, val_loader, val_size, device, criterion, optimizer, scheduler, num_epochs, workspace)

timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[0]
if os.path.isdir(os.path.join(workspace, 'checkpoints/best')) != True:
        os.mkdir(os.path.join(workspace, 'checkpoints/best'))
model_path = os.path.join(workspace, 'checkpoints/best', timestamp + '.pth')
torch.save(model.state_dict(), model_path)

utils.plot_loss_acc(timestamp, workspace, model_name, optimizer_name, epochs, train_losses, val_losses, train_accs, val_accs)
utils.create_train_log(workspace, train_accs, train_losses, train_f1_list, val_accs, val_losses, val_f1_list, model_name, optimizer_name, criterion_name, lr, momentum, step_size, gamma, num_epochs)






