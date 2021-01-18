# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:41:27 2020

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

def load_model(path, nets, num_classes):
    print('[INFO] loading model...')
    model_list = []
    model_name = []

    model_0 = models.resnext50_32x4d(pretrained=True)
    model_0.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model_0.fc.in_features, num_classes))

    model_1 = models.inception_v3(pretrained=True, aux_logits=False)
    model_1.fc = nn.Linear(model_1.fc.in_features, num_classes)

    model_2 = models.densenet161(pretrained=True)
    model_2.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model_2.classifier.in_features, num_classes))
    model_2 = torch.nn.DataParallel(model_2).module

    for m in nets:
        if 'densenet.pth' in m:
            model_2.load_state_dict(torch.load(m, map_location='cuda:0'))
            # model_2.to(device)
            model_name.append('DenseNet-161')
            print(m, 'loaded')
        elif 'inception.pth' in m:
            model_1.load_state_dict(torch.load(m, map_location='cuda:0'))
            # model_1.to(device)
            model_name.append('Inception_v3')
            print(m, 'loaded')
        elif 'resnext' in m:
            model_0.load_state_dict(torch.load(m, map_location='cuda:0'))
            # model_0.to(device)
            model_name.append('ResNeXt-50')
            print(m, 'loaded')

    model_list.extend([model_0, model_1, model_2])

    return model_list, model_name


def predict_with_ensemble(outs, labels):
    tot = np.zeros((len(outs[0]), outs[0][0].size))

    for matrix in outs:
        tot += matrix

    tot /= len(outs)

    _, preds = torch.max(torch.from_numpy(tot), 1)
    total_correct = torch.sum(preds == labels)

    return preds, labels, total_correct

def main():
    workspace = os.path.abspath("../")
    num_classes = 5
    trained = 'last'
    # test_fold = 1
    data_dir = os.path.join(workspace, 'Dataset' + str(num_classes))

    # ResNeXt and DenseNet
    transform0 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Inception
    transform1 = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, 'Val')

    test_set_0 = datasets.ImageFolder(test_dir, transform0)
    test_set_1 = datasets.ImageFolder(test_dir, transform1)

    test_size = len(test_set_0)
    classes = test_set_0.classes

    test_loader0 = torch.utils.data.DataLoader(test_set_0, batch_size=64, shuffle=False, num_workers=0)
    test_loader1 = torch.utils.data.DataLoader(test_set_1, batch_size=64, shuffle=False, num_workers=0)

    loaders = [test_loader0, test_loader1, test_loader0]

    model_path = os.path.join(workspace, 'checkpoints', trained + str(num_classes))
    nets = [os.path.join(model_path, x) for x in os.listdir(model_path)]

    model_list, model_name = load_model(model_path, nets, num_classes)

    test_accs, test_f1s, cms, outs = [], [], [], []

    softmax = nn.Softmax(dim=0)

    for i in range(len(model_list)):
        test_acc, f1_test, cm, out, labels = tester.test(model_list[i], device, loaders[i], test_size, model_name[i])

        out = [softmax(x).cpu().numpy() for x in out]
        out = np.asmatrix(out)

        test_accs.append(test_acc)
        test_f1s.append(f1_test)
        cms.append(cm)
        outs.append(out)

        utils.plot_confusion_matrix(cm, classes, model_name[i] + '_' + str(num_classes), workspace, model_name[i] + ' - Acc: ' + str(round(test_acc.item(), 3)) + '%', save=False)
        #utils.create_test_log(workspace, cm, test_acc, f1_test, model_name[i], test_fold)

    preds, labels, total_correct = predict_with_ensemble(outs, utils.list_toTorch(labels))

    # for i in range(len(preds)):
    #     if labels[i] == 1 and preds[i] != labels[i]:
    #         sample_fname, _ = test_loader0.dataset.samples[i]
    #         print(sample_fname.split('\\')[-1])
    #         print('pred', preds[i].item(),'label', labels[i].item())

    total_acc = total_correct.numpy() / len(preds.numpy())
    total_fscore = f1_score(labels, preds, average='micro')
    total_cm = confusion_matrix(labels, preds)

    print('\n[INFO] ensemble model testing complete')
    print('- total accuracy = ', total_acc)
    print('- total F1-score = ', total_fscore)

    #timestamp = str(datetime.datetime.now()).split('.')[0]
    utils.plot_confusion_matrix(total_cm, classes, 'ensamble_' + str(num_classes) + '_' + trained, workspace, 'Ensamble - acc: ' + str(round(total_acc.item(),3)) + '%', save=False)


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main()
