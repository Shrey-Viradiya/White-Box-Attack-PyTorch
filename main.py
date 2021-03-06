import torch
import torchvision
from train import ModelToBreak, train
from attack import Launch

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((150,150)),
        torchvision.transforms.ColorJitter(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=img_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                            shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=img_transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # train_data_path = "./data/Intel_classification/seg_train/seg_train/"
    # train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms)
    # val_data_path = "./data/Intel_classification/seg_test/seg_test/"
    # val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms)
    
    # batch_size=10
    # trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    # valloader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # classes = train_data.classes
    
    try:
        model = torch.load('./saved_models/best_model_2')
    except:
        model = ModelToBreak()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fun = torch.nn.CrossEntropyLoss()


    print("Strting Training.....\n")
    train(model, optimizer, loss_fun, trainloader, valloader, epochs = 10, device=device)
    print("Training Completed.....\n")

    model = torch.load('./saved_models/best_model_2')
    
    Launch(model, valloader, loss_fun, classes, device= device)    
