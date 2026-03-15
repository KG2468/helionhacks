#!POPCORN leaderboard fp8_quant
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import helion
import helion.language as hl

FP8_MAX = 448.0
FP8_EPS = 1e-10

@helion.kernel(static_shapes=True, autotune_effort="none")
def fp8_kernel(
    x: torch.Tensor,
    x_q: torch.Tensor,
    x_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_groups, group_size = x.size()
    gs = hl.specialize(group_size)

    for tile_g in hl.tile(total_groups):
        group = x[tile_g, :].to(torch.float32)
        absmax = torch.amax(torch.abs(group), dim=-1)
        absmax = torch.clamp(absmax, min=FP8_EPS)
        scale = absmax / FP8_MAX
        quantized = torch.clamp(group / scale[:, None], -FP8_MAX, FP8_MAX)
        x_q[tile_g, :] = quantized
        x_s[tile_g] = scale

    return x_q, x_s


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    num_tokens, hidden_dim = x.shape
    num_groups = x_s.shape[1]
    group_size = hidden_dim // num_groups
    total_groups = num_tokens * num_groups

    x_flat = x.reshape(total_groups, group_size).contiguous()
    xq_flat = x_q.reshape(total_groups, group_size).contiguous()
    xs_flat = x_s.reshape(total_groups).contiguous()

    fp8_kernel(x_flat, xq_flat, xs_flat)

    return x_q, x_s