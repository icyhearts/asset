from torch import nn
rnn = nn.LSTM(10, 20, 2)
print(f"rnn:{rnn}")

input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
print(f"input.shape:{input.shape}, h0.shape:{h0.shape}, c0.shape:{c0.shape}")
output, (hn, cn) = rnn(input, (h0, c0))
print(f"output.shape:{output.shape}, hn.shape:{hn.shape}, cn.shape:{cn.shape}")

