# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:16:01 2020

@author: Edoardo Vantaggiato
"""

import os
import datetime
import json
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import adabound
from matplotlib import pyplot as plt
from time import sleep
import datetime
import cv2
import shutil
from tqdm import tqdm


class TwoInputsNet(nn.Module):
    
  def __init__(self, model1, model2, num_classes):
      
    super(TwoInputsNet, self).__init__()
    
    # self.model1 = models.densenet161(pretrained=True)   
    # self.model1.classifier = nn.Sequential(
    # nn.Dropout(0.3),
    # nn.Linear(2208, 1024)
    # )

    # self.model2 = models.inception_v3(pretrained=True)
    # self.model2.fc = nn.Linear(2048, 1024)
    
    self.model1 = model1
    self.model2 = model2
    self.fc2 = nn.Linear(2048, num_classes)

  def forward(self, input1, input2):
      
    c = self.model1(input1)
    f = self.model2(input2)
        
    combined = torch.cat((c,f), dim=1)

    out = self.fc2(F.relu(combined))
    
    return out

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # print(self.gamma)
        targets = targets.type(torch.long)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean() 
        
def plot_loss_acc(timestamp, workspace, model_name, optimizer_name, epochs, train_losses, val_losses, train_accs, val_accs, img_size=[12,5]):
    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(img_size[0], img_size[1]))
    fig.suptitle(model_name + ' with ' + optimizer_name + ' optimizer')
    axs1.plot(epochs, train_losses, label='Training')
    axs1.plot(epochs, val_losses, label='Validation')
    axs1.set(xlabel='Epochs', ylabel='Loss')
    axs1.legend()
    axs2.plot(epochs, train_accs, label='Training')
    axs2.plot(epochs, val_accs, label='Validation')
    axs2.set(xlabel='Epochs', ylabel='Accuracy')
    axs2.legend()
    if os.path.isdir(os.path.join(workspace, 'graph')) != True:
      os.mkdir(os.path.join(workspace, 'graph'))
    fig.savefig(os.path.join(workspace, 'graph', timestamp + '.png'))   # save the figure to file
    # plt.show()
    
def plot_confusion_matrix(cm, classes, timestamp, workspace, model_name, cmap=plt.cm.Blues, save=True):
    plt.figure(figsize=(13,11))
    if os.path.isdir(os.path.join(workspace, 'graph')) != True and save == True:
      os.mkdir(os.path.join(workspace, 'graph'))        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(model_name)
    if save:
        cm_path = timestamp + '_cm.png'
        plt.savefig(os.path.join(workspace, 'graph', cm_path))
    # plt.show()
    
def create_train_log(workspace, train_accs, train_losses, train_f1_list, val_accs, val_losses, val_f1_list, model_name, optimizer_name, criterion_name, lr, momentum, step_size, gamma, num_epochs):
    save_path = os.path.join(workspace, "log")
    timestamp = str(datetime.datetime.now()).split('.')[0]
    log = json.dumps({
        'model': model_name,
        'optimizer': optimizer_name,
        'loss function': criterion_name,
        'timestamp': timestamp,
        'num_epoch': num_epochs,
        'lr': lr,
        'momentum': momentum,
        'step_size': step_size,
        'gamma': gamma,
        'last_train_acc': float('%.5f' % train_accs[-1]),
        'best_train_acc': float('%.5f' % max(train_accs)),
        'last_train_loss': float('%.5f' % train_losses[-1]),
        'best_train_loss': float('%.5f' % min(train_losses)),
        'last_train_f1': float('%.5f' % train_f1_list[-1]),
        'best_train_f1': float('%.5f' % max(train_f1_list)),
        'last_val_acc': float('%.5f' % val_accs[-1]),
        'best_val_acc': float('%.5f' % max(val_accs)),
        'last_val_loss': float('%.5f' % val_losses[-1]),
        'best_val_loss': float('%.5f' % min(val_losses)),
        'last_val_f1': float('%.5f' % val_f1_list[-1]),
        'best_val_f1': float('%.5f' % max(val_f1_list)),
        'train_accuracies': train_accs,
        'train_losses': train_losses,
        'train_f1_list': train_f1_list,
        'val_accuracies': val_accs,
        'val_losses': val_losses,
        'val_f1_list': val_f1_list
    }, ensure_ascii=False, indent=4)
    save_log(log, save_path)
    
def create_test_log(workspace, cm, test_acc, f1_test, model_name):
    save_path = os.path.join(workspace, "log")
    timestamp = str(datetime.datetime.now()).split('.')[0]
    log = json.dumps({
        'model': model_name,
        'timestamp': timestamp,
        'test_acc': float('%.5f' % test_acc),
        'test_f1': float('%.5f' % f1_test),
        'confusion matrix': cm.tolist()
    }, ensure_ascii=False, indent=4)
    save_log(log, save_path, train=False)
    
def save_log(log, save_path, train=True):
    timestamp = json.loads(log)['timestamp']
    if train:
        log_name = timestamp.split(' ')[0] + '.log'
    else:
        log_name = timestamp.split(' ')[0] + '_TEST.log'
    if os.path.isdir(save_path) != True:
        os.mkdir(save_path)
    log_file = os.path.join(save_path, log_name)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write("{}\n".format(log))
      
def transform_image(img):
    min_size = min(img.shape[0],img.shape[1])
    max_crop = min_size - 224       # 224 for ResNet50
    
    pil_transform = transforms.ToPILImage()
    resize_transform = transforms.Resize(224)
    
    total_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2),
            transforms.Pad((10,10))
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.RandomRotation(30),
        transforms.RandomCrop(min_size - round(max_crop/10))
    ])
    
    image = pil_transform(img)
    
    if min_size < 224:
        image = resize_transform(image)
    
    return total_transform(image)

# About RANDOMCROP transformation
# ResNet50 would a 224x224 sized images
# Due to differente size of images in dataset, random crop must preserve at least
# 224 pixels for each dimensions. With max_crop I obtain the maximum crop to preserve 224 pixels
# on minimum size. Then I crop min_size - max_crop/10
    
def data_augmentation(workspace, data_dir, source_dirs):
    augset_dir = os.path.join(workspace, 'Augmented_TrainSet')
    if os.path.isdir(augset_dir) != True:
        os.mkdir(augset_dir)
    for c in source_dirs:
        if (os.path.isdir(os.path.join(augset_dir, c)) != True):
            os.mkdir(os.path.join(augset_dir, c))
        imgs = [x for x in os.listdir(os.path.join(data_dir, c))]
        for i, img in enumerate(imgs):
            original_img = img
            source_path = os.path.join(data_dir, c, original_img)
            target_path = os.path.join(augset_dir, c)
            shutil.copy(source_path, target_path)
            img = cv2.imread(source_path)
            for j in range(12):
                new_img = np.array(transform_image(img))
                new_img_name = "{}_copy{}.{}".format("".join(original_img.split(".")[:-1]),j,original_img.split(".").pop(-1))
                cv2.imwrite(os.path.join(target_path, new_img_name), new_img)
                print("Immagine {} trasformazione {} salvata".format(i, j), end="\r")

def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g['lr']
    
def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
    return optimizer
    
def get_momentum(optimizer):
    for g in optimizer.param_groups:
        return g['momentum']
    
def set_momentum(optimizer, momentum):
    for g in optimizer.param_groups:
        g['momentum'] = momentum
    return optimizer

def list_toTorch(list):
    return torch.from_numpy(np.array(list))

def recall_model(model_name):
    if model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'ResNeXt50':
        model = models.resnext50_32x4d(pretrained=True)
    elif model_name == 'Inception_v3':
        model = models.inception_v3(pretrained=True, aux_logits=False)
    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=True)    
    return model

def edit_model(model_name, model, dropout, prob, freeze, num_classes):
    if freeze:
        print ("\n[INFO] Freezing feature layers...")    
        for param in model.parameters():
            param.requires_grade=False    
        sleep(0.5)
        print("-"*50)
    if model_name == 'DenseNet161':
        num_ftrs = model.classifier.in_features
        if dropout:
            model.classifier = nn.Sequential(
                nn.Dropout(prob),
                nn.Linear(num_ftrs, num_classes))
        else:
            model.classifier = nn.linea(num_ftrs, num_classes)
    elif model_name == 'Inception_v3':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        if dropout:
            model.fc = nn.Sequential(
                nn.Dropout(prob),
                nn.Linear(num_ftrs, num_classes)) 
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_transform(model_name):
    if model_name == 'Inception_v3':
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   
        ])
    return transform
    

def create_model(workspace, dataset, num_classes, model_name, freeze, dropout, prob, lr, momentum, step_size, gamma, criterion_name, optimizer_name, device, dataparallel, gpus):
    if model_name == 'ResNet50' or model_name == 'ResNeXt50' or model_name =='Inception_v3' or model_name == 'DenseNet161':
        model = recall_model(model_name)
        model = edit_model(model_name, model, dropout, prob, freeze, num_classes)
        transform = create_transform(model_name)
    elif len(model_name) == 2:
        transform = []
        model1 = recall_model(model_name[0])
        model1 = edit_model(model_name[0], model1, dropout, prob, freeze, 1024)
        transform.append(create_transform(model_name[0]))
        model2 = recall_model(model_name[1])
        model2 = edit_model(model_name[1], model2, dropout, prob, freeze, 1024)
        transform.append(create_transform(model_name[1]))
        model = TwoInputsNet(model1, model2, num_classes)      
    
    if dataparallel:
        model = torch.nn.DataParallel(model, device_ids=gpus)
    
    if optimizer_name == 'SGD':
        optimizer_conv = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'AdamW':
        optimizer_conv = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer_conv = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdaBound':
        optimizer_conv = adabound.AdaBound(model.parameters(), lr=lr)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
    
    if criterion_name == 'Cross Entropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'Focal':
        criterion = FocalLoss()
        
    return model, optimizer_conv, criterion, exp_lr_scheduler, transform

