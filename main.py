import numpy as np
import time
import torch
import torch.nn as nn
from train import ModelToBreak, train

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    
    model = ModelToBreak()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_fun = torch.nn.CrossEntropyLoss()

    train_data = (np.load('./data/train_images.npy'), np.load('./data/train_labels.npy'))
    test_data  = (np.load('./data/test_images.npy'), np.load('./data/test_labels.npy'))

    print("Strting Training.....\n")
    train(model, optimizer, loss_fun, train_data, test_data, device=device)
    print("Training Completed.....\n")

    torch.save(model, './saved_models/model')