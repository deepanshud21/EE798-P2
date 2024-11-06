import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Dual Encoder Model
class DualEncoder(nn.Module):
    def __init__(self):
        super(DualEncoder, self).__init__()
        # Define convolutional layers for encoding color (albedo) and shading
        self.albedo_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.shading_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        albedo_triplane = self.albedo_encoder(x)
        shading_triplane = self.shading_encoder(x)
        return albedo_triplane, shading_triplane

# Define Temporal Consistency Network
class TemporalConsistencyNetwork(nn.Module):
    def __init__(self):
        super(TemporalConsistencyNetwork, self).__init__()
        # Define a simple LSTM for temporal consistency
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        # x shape should be (batch_size, seq_len, feature_dim)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Final time step
        return output
