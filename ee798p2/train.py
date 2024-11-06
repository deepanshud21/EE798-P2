import torch
from torch import nn, optim
from data_loader import get_dataloader
from utils import albedo_loss, shading_loss, rgb_loss, adversarial_loss
from model import DualEncoder, TemporalConsistencyNetwork
from tqdm import tqdm
dual_encoder = DualEncoder()
temporal_consistency_net = TemporalConsistencyNetwork()
optimizer = optim.Adam(list(dual_encoder.parameters()) + list(temporal_consistency_net.parameters()), lr=1e-4)

train_loader = get_dataloader(batch_size=8, train=True)

for epoch in range(1, 21):
    dual_encoder.train()
    temporal_consistency_net.train()
    epoch_loss = 0
    for img_batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        albedo_triplane, shading_triplane = dual_encoder(img_batch)
        
        loss_albedo = albedo_loss(albedo_triplane, img_batch)
        loss_shading = shading_loss(shading_triplane, img_batch)
        loss_rgb = rgb_loss(albedo_triplane, shading_triplane, img_batch)
        loss_adv = adversarial_loss(albedo_triplane, img_batch)
        
        total_loss = loss_albedo + loss_shading + loss_rgb + loss_adv
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    
    print(f"Epoch [{epoch}], Loss: {epoch_loss / len(train_loader)}")
    
    torch.save(dual_encoder.state_dict(), f'checkpoints/dual_encoder_epoch_{epoch}.pt')
    torch.save(temporal_consistency_net.state_dict(), f'checkpoints/temporal_net_epoch_{epoch}.pt')
