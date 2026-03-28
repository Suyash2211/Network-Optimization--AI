import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
if __name__ == "__main__":
    model = DQN(input_dim=6, output_dim=4)

    sample_input = torch.randn(1, 6)
    output = model(sample_input)

    print("Q-values:", output)