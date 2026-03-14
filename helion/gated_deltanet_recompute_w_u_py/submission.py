#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch

CHUNK_SIZE = 64

def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = CHUNK_SIZE
    NT = T // C

    k_c = k.reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4).contiguous()
    v_c = v.reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4).contiguous()
    beta_c = beta.reshape(B, NT, C, H).permute(0, 1, 3, 2).contiguous()
    g_c = g.reshape(B, NT, C, H).permute(0, 1, 3, 2).contiguous()
    A_c = A.reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4).contiguous()

    beta_v = v_c * beta_c.unsqueeze(-1)
    beta_exp_g_k = k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1)

    BNH = B * NT * H
    A_flat = A_c.reshape(BNH, C, C)
    u_flat = torch.bmm(A_flat, beta_v.reshape(BNH, C, V))
    w_flat = torch.bmm(A_flat, beta_exp_g_k.reshape(BNH, C, K))

    w = w_flat.reshape(B, NT, H, C, K).permute(0, 1, 3, 2, 4).reshape(B, T, H, K)
    u = u_flat.reshape(B, NT, H, C, V).permute(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return w.to(k.dtype), u.to(v.dtype)
