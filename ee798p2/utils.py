import torch
import torch.nn.functional as F

def albedo_loss(pred, target):
    return F.l1_loss(pred, target)

def shading_loss(pred, target):
    return F.l1_loss(pred, target)

def rgb_loss(albedo, shading, target):
    return F.l1_loss(albedo * shading, target)

def adversarial_loss(pred, target):
    return -torch.mean(torch.log(pred) + torch.log(1 - target))

import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def render_final_output(predictions, albedo, shading, save_path="output"):
    import os
    os.makedirs(save_path, exist_ok=True)
    if predictions.requires_grad:
        predictions = predictions.detach().cpu()
    if albedo.requires_grad:
        albedo = albedo.detach().cpu()
    if shading.requires_grad:
        shading = shading.detach().cpu()

    combined_output = predictions * shading + albedo 
    save_image(predictions, f"{save_path}/predicted_output.png")
    save_image(albedo, f"{save_path}/albedo_output.png")
    save_image(shading, f"{save_path}/shading_output.png")
    save_image(combined_output, f"{save_path}/combined_output.png")
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    images = [predictions, albedo, shading, combined_output]
    titles = ["Prediction", "Albedo", "Shading", "Combined"]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Rendered images saved to {save_path}")

