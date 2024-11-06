import torch
import cv2
from model import DualEncoder, TemporalConsistencyNetwork
from utils import render_final_output

dual_encoder = DualEncoder()
temporal_consistency_net = TemporalConsistencyNetwork()
dual_encoder.load_state_dict(torch.load('checkpoints/dual_encoder_final.pt'))
temporal_consistency_net.load_state_dict(torch.load('checkpoints/temporal_net_final.pt'))

input_img = cv2.imread('sample_frame.png')
input_tensor = torch.tensor(input_img).float().unsqueeze(0)  # Add batch dimension

albedo_triplane, shading_triplane = dual_encoder(input_tensor)

viewing_angle = 30 
final_output = render_final_output(albedo_triplane, shading_triplane, viewing_angle)
cv2.imwrite('rendered_output.png', final_output)
