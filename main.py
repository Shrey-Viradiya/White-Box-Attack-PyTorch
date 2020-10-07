import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from train import ModelToBreak, train
import matplotlib.pyplot as plt

def fgsm(input_tensor, labels, loss_function, model, epsilon=0.02):
    outputs = model(input_tensor)
    loss = loss_function(outputs, labels)
    loss.backward(retain_graph=True)
    vals = torch.sign(input_tensor.grad) * epsilon
    return vals

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
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model = ModelToBreak()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    loss_fun = torch.nn.CrossEntropyLoss()


    print("Strting Training.....\n")
    # train(model, optimizer, loss_fun, trainloader, testloader, device=device)
    print("Training Completed.....\n")

    model = torch.load('./saved_models/best_model')
    model.to(device)

    image, label = next(iter(testloader))
    image = image.to(device)
    image.requires_grad = True
    label = label.to(device)

    adversarial_mask = fgsm(image, label, loss_fun,  model, epsilon=0.5)

    real_pred = model(image).max(1, keepdim=True)[1].item()
    adve_pred = model(adversarial_mask).max(1, keepdim=True)[1].item()
    print(f"Real Label: {label.item()}, Predicted Label: {real_pred}, Adversarial Mask Prediction: {adve_pred}")
    
    title = f"Real Label: {classes[label.item()]}, Predicted Label: {classes[real_pred]}, Adversarial Mask Prediction: {classes[adve_pred]}"
    print(title)

    # plot the images

    plt.figure(figsize = (4,2))

    plt.suptitle(title)
    plt.subplot(1,2,1)
    plt.imshow(image.squeeze().permute(1,2,0).cpu().detach().numpy())
    plt.axis(False)

    plt.subplot(1,2,2)
    plt.imshow(adversarial_mask.squeeze().permute(1,2,0).cpu().detach().numpy())
    plt.axis(False)

    plt.savefig('./output.png', dpi = 300)
    