# -*- coding: utf-8 -*-
"""
Created on Mon Oct 8 11:27:15 2020

@author: Edoardo Vantaggiato
"""

import torch
from sklearn.metrics import f1_score

import os
import time
from time import sleep
import copy
from tqdm import tqdm
import datetime

import Utils as utils

def train(model, train_loader, train_size, val_loader, val_size, device, criterion, optimizer, scheduler, num_epochs, workspace):
    
    epochs = []
    train_accs, train_losses, train_f1_list = [], [], []
    val_accs, val_losses, val_f1_list = [], [], []
    
    print("[INFO] Training started")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
        
    for epoch in range(num_epochs):
        epochs.append(epoch)
        print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
        sleep(0.2)
        
        # --- Switch to train mode --- #
        model.train()
        running_loss = 0.0
        running_corrects = 0
        t_train = tqdm(train_loader)
        l = 0
        tot_labels = []
        tot_preds = []
        
        for inputs, labels in t_train:
            l = l + len(inputs)
            t_train.set_description("Train {}/{}".format(l, train_size), True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                # _ contains max value for each row of outputs
                # preds contains index of the predicted class
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
    
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                
            for x in labels:
                tot_labels.append(x.item())
            for y in preds:
                tot_preds.append(y.item())
                
            f1_train = f1_score(tot_labels, tot_preds, average='weighted')
    
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            t_train.set_postfix({'Train loss': running_loss / l, 'Train accu': (running_corrects.double() / l).item(), 'F1-score': f1_train}, True)
        
        scheduler.step()
        train_loss = running_loss / train_size  # modify to run also in validation phase 
        train_acc = running_corrects.double() / train_size  # modify to run also in validation phase
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())
        train_f1_list.append(f1_train)
        
        # torch.cuda.empty_cache()
        
        # --- Switch to evaluation mode --- #
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        t_val = tqdm(val_loader)
        l = 0
        tot_labels = []
        tot_preds = []
        
        for inputs, labels in t_val:
            l = l + len(inputs)
            t_val.set_description("Validation {}/{}".format(l, val_size), True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            for x in labels:
                tot_labels.append(x.item())
            for y in preds:
                tot_preds.append(y.item())
                
            f1_val = f1_score(tot_labels, tot_preds, average='micro')
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            t_val.set_postfix({'Val loss': running_loss / l, 'Val accu': (running_corrects.double() / l).item(), 'F1-score': f1_val}, True)
        
        val_loss = running_loss / val_size  # modify to run also in validation phase 
        val_acc = running_corrects.double() / val_size  # modify to run also in validation phase
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        val_f1_list.append(f1_val)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if epoch == num_epochs - 1:
            timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[0]
            if os.path.isdir(os.path.join(workspace, 'checkpoints/last')) != True:
                os.mkdir(os.path.join(workspace, 'checkpoints/last'))
            model_path = os.path.join(workspace, 'checkpoints/last', timestamp + '_epoch-' + str(epoch+1) + '.pth' )
            torch.save(model.state_dict(), model_path)    
        
    # load best model weights
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print('\n[INFO] Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Training\n- max accuracy = {}\n- min loss = {}\n- max F1-score = {}'.format(max(train_accs), min(train_losses), max(train_f1_list)))
    print('Validation\n- max accuracy = {}\n- min loss = {}\n- max F1-score = {}\n'.format(max(val_accs), min(val_losses), max(val_f1_list)))
    
    return model, epochs, train_accs, train_losses, val_accs, val_losses, train_f1_list, val_f1_list


def trainTwoNet(model, train_loader1, train_loader2, train_size, val_loader1, val_loader2, val_size, device, criterion, optimizer, scheduler, num_epochs, workspace, s):
    epochs = []
    train_accs, train_losses, train_f1_list = [], [], []
    val_accs, val_losses, val_f1_list = [], [], []
    
    print("[INFO] Training started")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
        
    for epoch in range(num_epochs):
        epochs.append(epoch)
        print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
        sleep(0.2)
        
        # --- Switch to train mode --- #
        model.train()
        running_loss = 0.0
        running_corrects = 0
        t_train = tqdm(zip(train_loader1, train_loader2))
        l = 0
        tot_labels = []
        tot_preds = []
        
        for (inputs1, labels1), (inputs2, labels2) in t_train:
            l = l + len(inputs1)
            t_train.set_description("Train {}/{}".format(l, train_size), True)
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels1 = labels1.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs1, inputs2)
                # _ contains max value for each row of outputs
                # preds contains index of the predicted class
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels1)
    
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                
            for x in labels1:
                tot_labels.append(x.item())
            for y in preds:
                tot_preds.append(y.item())
                
            f1_train = f1_score(tot_labels, tot_preds, average='weighted')
    
            # statistics
            running_loss += loss.item() * inputs1.size(0)
            running_corrects += torch.sum(preds == labels1)
            t_train.set_postfix({'Train loss': running_loss / l, 'Train accu': (running_corrects.double() / l).item(), 'F1-score': f1_train}, True)
            
        if str(optimizer).split(' (')[0] != 'AdaBound':
            scheduler.step()
        train_loss = running_loss / train_size  # modify to run also in validation phase 
        train_acc = running_corrects.double() / train_size  # modify to run also in validation phase
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())
        train_f1_list.append(f1_train)
        
        # torch.cuda.empty_cache()
        
        # --- Switch to evaluation mode --- #
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        t_val = tqdm(zip(val_loader1, val_loader2))
        l = 0
        tot_labels = []
        tot_preds = []
        
        for (inputs1, labels1), (inputs2, labels2) in t_val:
            l = l + len(inputs1)
            t_val.set_description("Validation {}/{}".format(l, val_size), True)
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels1 = labels1.to(device)
            
            with torch.set_grad_enabled(False):
                optimizer.zero_grad()
                outputs = model(inputs1, inputs2)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels1)
            
            for x in labels1:
                tot_labels.append(x.item())
            for y in preds:
                tot_preds.append(y.item())
                
            f1_val = f1_score(tot_labels, tot_preds, average='micro')
            
            running_loss += loss.item() * inputs1.size(0)
            running_corrects += torch.sum(preds == labels1)
            t_val.set_postfix({'Val loss': running_loss / l, 'Val accu': (running_corrects.double() / l).item(), 'F1-score': f1_val}, True)
        
        val_loss = running_loss / val_size  # modify to run also in validation phase 
        val_acc = running_corrects.double() / val_size  # modify to run also in validation phase
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        val_f1_list.append(f1_val)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        #if epoch == num_epochs-1:
        if not (epoch+1) % 10:
            timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[0]
            if os.path.isdir(os.path.join(workspace, 'checkpoints/last')) != True:
                os.mkdir(os.path.join(workspace, 'checkpoints/last'))
            model_path = os.path.join(workspace, 'checkpoints/last', timestamp + '_epoch-' + str(epoch+1) + '_test-' + str(s[0]) + '.pth' )
            torch.save(model.state_dict(), model_path)    
        
    # load best model weights
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print('\n[INFO] Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Training\n- max accuracy = {}\n- min loss = {}\n- max F1-score = {}'.format(max(train_accs), min(train_losses), max(train_f1_list)))
    print('Validation\n- max accuracy = {}\n- min loss = {}\n- max F1-score = {}\n'.format(max(val_accs), min(val_losses), max(val_f1_list)))
    
    return model, epochs, train_accs, train_losses, val_accs, val_losses, train_f1_list, val_f1_list
    
