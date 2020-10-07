import matplotlib.pyplot as plt
import numpy as np
import torch


def fgsm(input_tensor, labels, loss_function, model, epsilon=0.02):
    outputs = model(input_tensor)
    loss = loss_function(outputs, labels)
    loss.backward(retain_graph=True)
    vals = torch.sign(input_tensor.grad) * epsilon
    return vals

def Launch(model, testloader, loss_fun, classes, device = 'cuda'):
    """
    docstring
    """
    model.to(device)
    
    i = 0

    for image, label in testloader:        
        image = image.to(device)
        image.requires_grad = True
        label = label.to(device)

        epsilon = 0.2
        adversarial_mask = fgsm(image, label, loss_fun,  model, epsilon=epsilon)

        real_label = classes[label.item()]
        real_pred = classes[model(image).max(1, keepdim=True)[1].item()]
        adve_pred = classes[model(adversarial_mask).max(1, keepdim=True)[1].item()]

        if real_label != real_pred:
            continue

        title = f"Real Label: {real_label}, Predicted Label: {real_pred}, Adversarial Mask Prediction: {adve_pred}"
        print(title)

        # plot the images

        plt.figure(figsize = (8,4))

        plt.suptitle(title)
        plt.subplot(1,2,1)
        image = image.squeeze().permute(1,2,0).cpu().detach().numpy()
        plt.imshow(image.astype(np.float64))
        plt.axis(False)

        plt.subplot(1,2,2)
        adversarial_mask = adversarial_mask.squeeze().permute(1,2,0).cpu().detach().numpy()
        plt.imshow(adversarial_mask.astype(np.float64))
        plt.axis(False)
        plt.tight_layout()
        plt.savefig(f'./outputs/output_{real_label}_{adve_pred}_{i}.png')
        plt.close('all')
        i+=1
        if i == 15:
            break