import torch
from power_iteration import PowerSVDLayer
input = torch.randn(5,5)
layer = PowerSVDLayer(3,input.shape,3)
output = layer(input)