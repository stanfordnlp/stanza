import torch
import torch.nn as nn
import torch.nn.functional as F

class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size, input2_size, output_size)
        self.W1 = nn.Linear(input1_size, output_size, bias=False)
        self.W2 = nn.Linear(input2_size, output_size, bias=False)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()
        self.W1.weight.data.zero_()
        self.W2.weight.data.zero_()

    def forward(self, input1, input2):
        res = self.W1(input1) + self.W2(input2) + self.W_bilin(input1, input2)
        return res

class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))), self.dropout(self.hidden_func(self.W2(input2))))

if __name__ == "__main__":
    x1 = torch.randn(3,4)
    x2 = torch.randn(3,5)
    scorer = DeepBiaffineScorer(4, 5, 6, 7)
    print(scorer(x1, x2))
