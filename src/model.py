import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Nature DQN CNN for 84x84 inputs and frame-stacked channels.
    Input: (B, C, 84, 84) with uint8 pixels [0..255]
    Output: Q-values per action (B, n_actions)
    """
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # For 84x84 after above convs, flattened size is 3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # Kaiming init for ReLU layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # convert to float and normalize to [0,1]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)      # (B, 3136)
        x = F.relu(self.fc1(x))
        return self.fc2(x)             # (B, n_actions)

