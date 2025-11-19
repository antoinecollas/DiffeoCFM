import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        time_emb_dim: int = 1,
        hidden_dim: list[int] = [128],
        dtype=torch.float64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(
            nn.Linear(input_dim + cond_dim + time_emb_dim, hidden_dim[0], dtype=dtype)
        )
        layers.append(nn.SELU())
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i], dtype=dtype))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim[-1], input_dim, dtype=dtype))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        t = t.view(-1, self.time_emb_dim).expand(x.shape[0], self.time_emb_dim)
        h = torch.cat([x, cond, t], dim=-1).to(dtype=self.net[0].weight.dtype)
        return self.net(h)
