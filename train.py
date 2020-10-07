import pickle
import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ModelToBreak(nn.Module):
    def __init__(self):
        super(ModelToBreak, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, optimizer, loss_fun, train_data ,test_data, epochs = 20, device = 'cuda'):
    '''
    Train function:

    parameters:

    model       : PyTorch Model
    optimizer   : optimizer object
    loss_fun    : Loss Function object
    train_data  : CIFAR10 train dataloader
    test_data   : CIFAR10 test  dataloader
    batch_size  : default value 100
    epochs      : default value 20
    device      : 'cuda' or 'cpu', default 'cuda'
    '''


    for epoch in range(epochs):
        start = time.time()

        training_loss = 0.0
        valid_loss = 0.0

        model.train()
        correct = 0 
        total = 0
        for batch in train_data:
            train_images, train_labels = batch
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()
            output = model(train_images)
            loss = loss_fun(output, train_labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += train_labels.size(0)
            correct += (predicted == train_labels).sum().item()
        training_accuracy = correct/total * 100

        model.eval()
        correct = 0 
        total = 0
        with torch.no_grad():
            for batch in test_data:
                test_images, test_labels = batch
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                output = model(test_images)
                loss = loss_fun(output,test_labels) 
                valid_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        testing_accuracy = correct/total * 100
        print(f'{bcolors.OKGREEN}Epoch:{bcolors.ENDC} {epoch}, {bcolors.OKGREEN}Training Loss:{bcolors.ENDC} {training_loss:.5f}, {bcolors.OKGREEN}Validation Loss:{bcolors.ENDC} {valid_loss:.5f}, {bcolors.OKGREEN}Training accuracy:{bcolors.ENDC} {training_accuracy:.2f}, {bcolors.OKGREEN}Testing accuracy:{bcolors.ENDC} {testing_accuracy:.2f}, {bcolors.OKGREEN}time:{bcolors.ENDC} {time.time() - start:.2f} s')
