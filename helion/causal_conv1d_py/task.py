from typing import TypeAlias

import torch

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
output_t: TypeAlias = torch.Tensor
