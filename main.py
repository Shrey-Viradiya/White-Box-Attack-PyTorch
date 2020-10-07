import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from train import ModelToBreak, train

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
                                    [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model = ModelToBreak()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    loss_fun = torch.nn.CrossEntropyLoss()


    print("Strting Training.....\n")
    train(model, optimizer, loss_fun, trainloader, testloader, device=device)
    print("Training Completed.....\n")

    best_model = torch.load('./saved_models/best_model')