import torch
import matplotlib.pyplot as plt
import cv2
from model import DualEncoder, TemporalConsistencyNetwork
from utils import render_final_output
import numpy as np

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.show()

def render_sample_outputs(model, sample_image_path, viewing_angles):
    dual_encoder = model['dual_encoder']
    sample_img = cv2.imread(sample_image_path)
    input_tensor = torch.tensor(sample_img).float().unsqueeze(0)  # Add batch dimension

    albedo_triplane, shading_triplane = dual_encoder(input_tensor)
    rendered_images = []
    for angle in viewing_angles:
        rendered_image = render_final_output(albedo_triplane, shading_triplane, angle)
        rendered_images.append(rendered_image)
    
    fig, axes = plt.subplots(1, len(rendered_images), figsize=(15, 5))
    for ax, img, angle in zip(axes, rendered_images, viewing_angles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Viewing Angle: {angle}')
    plt.savefig('rendered_sample_outputs.png')
    plt.show()

def evaluate_temporal_consistency(model, video_path):
    dual_encoder = model['dual_encoder']
    temporal_net = model['temporal_net']
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(torch.tensor(frame).float().unsqueeze(0))  # Add batch dimension
    
    cap.release()

    albedo_triplanes = []
    shading_triplanes = []
    for frame in video_frames:
        albedo, shading = dual_encoder(frame)
        albedo_triplanes.append(albedo)
        shading_triplanes.append(shading)

    albedo_triplanes = torch.stack(albedo_triplanes, dim=1)  # (batch, seq, features)
    shading_triplanes = torch.stack(shading_triplanes, dim=1)
    
    albedo_temporal_consistency = temporal_net(albedo_triplanes)
    shading_temporal_consistency = temporal_net(shading_triplanes)

    temporal_diff = torch.mean(torch.abs(albedo_temporal_consistency[:, 1:] - albedo_temporal_consistency[:, :-1]))
    print(f"Temporal consistency difference (Albedo): {temporal_diff.item()}")

    return temporal_diff.item()
