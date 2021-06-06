import torch.nn as nn
import torch.nn.functional as f


class MLP(nn.Module):

    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.input_layer_1 = nn.Linear(input_shape, hidden_shape)
        self.hidden_layer_2 = nn.Linear(hidden_shape, hidden_shape)
        self.hidden_layer_3 = nn.Linear(hidden_shape, hidden_shape)
        self.output_layer_4 = nn.Linear(hidden_shape, output_shape)

    def forward(self, input):
        x = f.relu(self.input_layer_1(input))
        x = f.relu(self.hidden_layer_2(x))
        x = f.relu(self.hidden_layer_3(x))
        output = self.output_layer_4(x)
        return output.squeeze()

