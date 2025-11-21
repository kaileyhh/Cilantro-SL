import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, primary_in_features, cond_in_features):
        super().__init__()
        # one linear layer to generate gamma and beta from the conditioning input.

        # gamma: shifting vector
        # beta: scaling vector
        self.cond_gen = nn.Linear(cond_in_features, primary_in_features * 2)
        
    def forward(self, primary_input, cond_input):
        # generate concatenated gamma and beta vector.
        cond_output = self.cond_gen(cond_input)
        
        # split output into gamma and beta.
        gamma, beta = torch.chunk(cond_output, 2, dim = -1)
        
        # apply the FiLM transformation: output = gamma * primary_input + beta
        # gamma(z) * x + beta(z)
        return primary_input * gamma + beta