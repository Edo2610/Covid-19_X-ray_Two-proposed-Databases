# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:57:54 2020

@author: Edoardo Vantaggiato
"""

import torch
import torch.nn as nn
import os
import numpy as np
from torchvision import transforms, datasets, models
from sklearn.metrics import f1_score, confusion_matrix
import datetime

import Tester as tester
import Utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name = 'densenet'
trained = 'last'
num_classes = 3


if name == 'densenet' or name == 'resnext':
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
else:
    transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

workspace = os.path.abspath("../")
data_dir = os.path.join(workspace, 'Dataset' + str(num_classes))

test_fold = 1
test_dir = os.path.join(data_dir, 'Test')

test_set = datasets.ImageFolder(test_dir, transform)

test_size = len(test_set)
classes = test_set.classes

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

if name == 'densenet':
    model = models.densenet161(pretrained=False)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier.in_features, num_classes))
    model = torch.nn.DataParallel(model).module
    model_path =  os.path.join(workspace, 'checkpoints', trained + str(num_classes), 'densenet.pth')
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
elif name == 'inception':
    model = models.inception_v3(pretrained=True, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model_path = os.path.join(workspace, 'checkpoints', trained + str(num_classes),'inception.pth')
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
else:
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    model_path = os.path.join(workspace, 'checkpoints', trained + str(num_classes), 'resnext.pth')
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

test_acc, f1_test, cm, out, labels = tester.test(model, device, test_loader, test_size, name)

utils.plot_confusion_matrix(cm, classes, name + '_' + str(num_classes) + '_' + trained, workspace, name + ' - acc: ' + str(test_acc.item()), save=True)
