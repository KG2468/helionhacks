#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_stages=4, num_warps=8),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_stages=1, num_warps=16),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_stages=8, num_warps=8),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_stages=2, num_warps=16),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_stages=6, num_warps=16),
    # Ranked shapes
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=16, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=16, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=16, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=16, num_stages=2),
}


def _make_fused_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def fused_kernel(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)
        NT = T // C
        BNH = B * NT * H

        w_out = torch.empty(B, T, H, K, dtype=k.dtype, device=k.device)
        u_out = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)

        for flat, tv in hl.tile([BNH, V], block_size=[1, 64]):
            idx = flat.begin
            i_b = idx // (NT * H)
            i_nt = (idx % (NT * H)) // H
            i_h = idx % H
            ts = i_nt * C

            b_A = A[i_b, ts:ts + C, i_h, :]
            b_beta = beta[i_b, ts:ts + C, i_h].to(torch.float32)

            b_v = v[i_b, ts:ts + C, i_h, tv]
            b_v_scaled = b_v * b_beta[:, None]
            b_u = hl.dot(b_A, b_v_scaled, out_dtype=torch.float32)
            u_out[i_b, ts:ts + C, i_h, tv] = b_u.to(v.dtype)

            b_k = k[i_b, ts:ts + C, i_h, tv]
            b_g = g[i_b, ts:ts + C, i_h].to(torch.float32)
            b_k_scaled = b_k * (b_beta * torch.exp(b_g))[:, None]
            b_w = hl.dot(b_A, b_k_scaled, out_dtype=torch.float32)
            w_out[i_b, ts:ts + C, i_h, tv] = b_w.to(k.dtype)

        return w_out, u_out

    return fused_kernel


SEPARATE_U_CONFIGS: dict[tuple, helion.Config] = {
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=1),
}

SEPARATE_W_CONFIGS: dict[tuple, helion.Config] = {
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=1),
}


def _make_u_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def u_kernel(
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        B, T, H, V = v.shape
        C = 64
        V = hl.specialize(V)
        NT = T // C
        BNH = B * NT * H
        u_out = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)

        for flat, tv in hl.tile([BNH, V], block_size=[1, 64]):
            idx = flat.begin
            i_b = idx // (NT * H)
            i_nt = (idx % (NT * H)) // H
            i_h = idx % H
            ts = i_nt * C

            b_A = A[i_b, ts:ts + C, i_h, :]
            b_v = v[i_b, ts:ts + C, i_h, tv]
            b_beta = beta[i_b, ts:ts + C, i_h].to(torch.float32)
            b_u = hl.dot(b_A, b_v * b_beta[:, None], out_dtype=torch.float32)
            u_out[i_b, ts:ts + C, i_h, tv] = b_u.to(v.dtype)

        return u_out

    return u_kernel


def _make_w_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def w_kernel(
        k: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        B, T, H, K = k.shape
        C = 64
        K = hl.specialize(K)
        NT = T // C
        BNH = B * NT * H
        w_out = torch.empty(B, T, H, K, dtype=k.dtype, device=k.device)

        for flat, tk in hl.tile([BNH, K], block_size=[1, 64]):
            idx = flat.begin
            i_b = idx // (NT * H)
            i_nt = (idx % (NT * H)) // H
            i_h = idx % H
            ts = i_nt * C

            b_A = A[i_b, ts:ts + C, i_h, :]
            b_k = k[i_b, ts:ts + C, i_h, tk]
            b_beta = beta[i_b, ts:ts + C, i_h].to(torch.float32)
            b_g = g[i_b, ts:ts + C, i_h].to(torch.float32)
            b_w = hl.dot(b_A, b_k * (b_beta * torch.exp(b_g))[:, None], out_dtype=torch.float32)
            w_out[i_b, ts:ts + C, i_h, tk] = b_w.to(k.dtype)

        return w_out

    return w_kernel


_FUSED = {s: _make_fused_kernel(c) for s, c in FUSED_CONFIGS.items()}
_U_SEP = {s: _make_u_kernel(c) for s, c in SEPARATE_U_CONFIGS.items()}
_W_SEP = {s: _make_w_kernel(c) for s, c in SEPARATE_W_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    key = (B, T, H, K, V)

    if key in _FUSED:
        return _FUSED[key](k, v, beta, A, g)

    if key in _U_SEP:
        u = _U_SEP[key](v, beta, A)
        w = _W_SEP[key](k, beta, A, g)
        return w, u

    C = 64
    NT = T // C
    BNH = B * NT * H
    k_c = k.reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4).reshape(BNH, C, K).contiguous()
    v_c = v.reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4).reshape(BNH, C, V).contiguous()
    beta_c = beta.reshape(B, NT, C, H).permute(0, 1, 3, 2).reshape(BNH, C).contiguous()
    g_c = g.reshape(B, NT, C, H).permute(0, 1, 3, 2).reshape(BNH, C).contiguous()
    A_c = A.reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4).reshape(BNH, C, C).contiguous()
    u_flat = torch.bmm(A_c, v_c * beta_c.unsqueeze(-1))
    w_flat = torch.bmm(A_c, k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1))
    w = w_flat.reshape(B, NT, H, C, K).permute(0, 1, 3, 2, 4).reshape(B, T, H, K)
    u = u_flat.reshape(B, NT, H, C, V).permute(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return w.to(k.dtype), u.to(v.dtype)
