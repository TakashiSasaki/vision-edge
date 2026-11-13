"""Exportable PyTorch modules for the remaining Mirage pipeline components:
  1. MirageReader: convolutional + gated-mix CTC text strip reader
  2. MirageCrossKv: region memory + 8-layer cross-attention key/value precompute
  3. MirageDecoderStep: single-step autoregressive mHC + Engram + GQA + CrossAttn + HadamardMLP decoder
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6
D_MODEL = 512
MAX_SLOTS = 96
MAX_M = 256 + MAX_SLOTS  # 352
MAX_SEQ = 516
ENGRAM_SITES = (3, 7)


def zc_norm(x: torch.Tensor, scale: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shape = [1] * x.dim()
    shape[dim] = -1
    return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + EPS) * (1.0 + scale.view(shape))


def rms_unit(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)


def conv3_sym(x: torch.Tensor, w_hwio: torch.Tensor, stride: tuple[int, int]) -> torch.Tensor:
    w = w_hwio.permute(3, 2, 0, 1).contiguous()
    return F.conv2d(F.pad(x, (1, 1, 1, 1)), w, stride=stride)


def dwconv_sym(x: torch.Tensor, w_hw1c: torch.Tensor, stride: tuple[int, int]) -> torch.Tensor:
    w = w_hw1c.permute(3, 2, 0, 1).contiguous()
    return F.conv2d(F.pad(x, (1, 1, 1, 1)), w, stride=stride, groups=x.shape[1])


def rope_half_step(x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = rope_cos.view(1, half)
    s = rope_sin.view(1, half)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def sinkhorn_4x4(logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
    k = logits
    for _ in range(iters):
        m_row = k.amax(dim=-1, keepdim=True)
        k = k - (m_row + torch.log(torch.exp(k - m_row).sum(dim=-1, keepdim=True)))
        m_col = k.amax(dim=-2, keepdim=True)
        k = k - (m_col + torch.log(torch.exp(k - m_col).sum(dim=-2, keepdim=True)))
    return torch.exp(k)


def _stack_layers(xs: list[torch.Tensor]) -> torch.Tensor:
    """Tree-structured stack keeping each ONNX Concat <= 4 inputs (<= 5 storage buffers for WebGPU)."""
    if len(xs) <= 4:
        return torch.stack(xs, dim=0)
    mid = len(xs) // 2
    return torch.cat([_stack_layers(xs[:mid]), _stack_layers(xs[mid:])], dim=0)


def build_sincos_2d_table() -> torch.Tensor:
    rows = torch.arange(256, dtype=torch.float32) // 16
    cols = torch.arange(256, dtype=torch.float32) % 16

    def one(pos: torch.Tensor, dd: int) -> torch.Tensor:
        omega = 1.0 / (10000.0 ** (torch.arange(dd // 2, dtype=torch.float32) / (dd // 2)))
        ang = pos[:, None] * omega[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    return torch.cat([one(rows, D_MODEL // 2), one(cols, D_MODEL // 2)], dim=-1)


def compute_rope_tables(max_seq: int = MAX_SEQ, head_dim: int = 48, theta: float = 100000.0) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq, dtype=torch.float32)[:, None]
    ang = pos * freqs[None, :]
    return ang.cos().contiguous(), ang.sin().contiguous()


def build_fourier_box(box: torch.Tensor) -> torch.Tensor:
    r = torch.arange(8, dtype=torch.float32)
    freqs = (2.0 ** r) * math.pi
    parts = []
    for c in range(4):
        a = freqs * float(box[c])
        parts.extend([a.sin(), a.cos()])
    return torch.cat(parts, dim=0)


class MirageReader(nn.Module):
    """Exportable Mirage CTC text strip reader: `[B, 3, 48, 320]` -> `[B, 80, 307]` probabilities."""

    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer('w0', weights['reader/w0'].clone().float())
        self.register_buffer('stage_norm_0', weights['reader/stage_norm_0/scale'].clone().float())
        for j in range(3):
            self.register_buffer(f'dw_{j}', weights[f'reader/dw_{j}'].clone().float())
            self.register_buffer(f'pw_{j}', weights[f'reader/pw_{j}/kernel'].clone().float())
            self.register_buffer(f'stage_norm_{j + 1}', weights[f'reader/stage_norm_{j + 1}/scale'].clone().float())
        for b in range(4):
            self.register_buffer(f'norm_{b}', weights[f'reader/norm_{b}/scale'].clone().float())
            self.register_buffer(f'mix_{b}', weights[f'reader/mix_{b}'][0, :, 0, :].clone().float())
            self.register_buffer(f'up_{b}', weights[f'reader/up_{b}/kernel'].clone().float())
            self.register_buffer(f'down_{b}', weights[f'reader/down_{b}/kernel'].clone().float())
        self.register_buffer('final_norm', weights['reader/final_norm/scale'].clone().float())
        self.register_buffer('out_kernel', weights['reader/out/kernel'].clone().float())
        self.register_buffer('out_bias', weights['reader/out/bias'].clone().float())

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        g = zc_norm(conv3_sym(crop, self.w0, (2, 2)), self.stage_norm_0, dim=1)
        strides = ((2, 2), (2, 1), (2, 1))
        for j, stride in enumerate(strides):
            y = dwconv_sym(F.silu(g), getattr(self, f'dw_{j}'), stride)
            pw_out = (y.permute(0, 2, 3, 1) @ getattr(self, f'pw_{j}')).permute(0, 3, 1, 2)
            g = zc_norm(pw_out, getattr(self, f'stage_norm_{j + 1}'), dim=1)
        z = F.silu(g).mean(dim=2).transpose(1, 2)  # [B, 80, 384]
        for b in range(4):
            h = zc_norm(z, getattr(self, f'norm_{b}'), dim=-1)
            mix = getattr(self, f'mix_{b}')  # [5, 384]
            h_pad = F.pad(h, (0, 0, 2, 2))
            h_mix = (
                h_pad[:, 0:80, :] * mix[0:1, :].unsqueeze(0)
                + h_pad[:, 1:81, :] * mix[1:2, :].unsqueeze(0)
                + h_pad[:, 2:82, :] * mix[2:3, :].unsqueeze(0)
                + h_pad[:, 3:83, :] * mix[3:4, :].unsqueeze(0)
                + h_pad[:, 4:84, :] * mix[4:5, :].unsqueeze(0)
            )
            vg = h_mix @ getattr(self, f'up_{b}')
            z = z + (vg[..., :384] * F.silu(vg[..., 384:])) @ getattr(self, f'down_{b}')
        logits = zc_norm(z, self.final_norm, dim=-1) @ self.out_kernel + self.out_bias
        return logits.softmax(dim=-1)


class MirageCrossKv(nn.Module):
    """Exportable Mirage region-memory + 8-layer cross-attention KV projection."""

    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer('sincos_2d', build_sincos_2d_table())
        self.register_buffer('pe_gate', weights['pe_gate'].clone().float())
        self.register_buffer('slot_proj', weights['slot_proj/kernel'].clone().float())
        self.register_buffer('slot_pool', weights['slot_pool/kernel'].clone().float())
        self.register_buffer('slot_norm', weights['slot_norm/scale'].clone().float())
        for l in range(8):
            self.register_buffer(f'cross_k_proj_{l}', weights['stack/layers/block/cross_attn/k_proj/kernel'][l].clone().float())
            self.register_buffer(f'cross_k_norm_{l}', weights['stack/layers/block/cross_attn/k_norm/scale'][l].clone().float())
            self.register_buffer(f'cross_v_proj_{l}', weights['stack/layers/block/cross_attn/v_proj/kernel'][l].clone().float())

    def forward(
        self,
        E: torch.Tensor,
        slot_d_fourier: torch.Tensor,
        slot_pooled: torch.Tensor,
        slot_emb: torch.Tensor,
        valid_cells: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cells = E[0] + torch.sigmoid(self.pe_gate) * self.sincos_2d  # [256, 512]
        z = slot_d_fourier[0] @ self.slot_proj + slot_pooled[0] @ self.slot_pool
        slots = zc_norm(z, self.slot_norm, dim=-1) + slot_emb[0]  # [96, 512]
        m = torch.cat([cells, slots], dim=0)  # [352, 512]
        ck_layers = []
        cv_layers = []
        for l in range(8):
            k = (m @ getattr(self, f'cross_k_proj_{l}')).view(MAX_M, 8, 48).transpose(0, 1)
            k = zc_norm(k, getattr(self, f'cross_k_norm_{l}'), dim=-1)
            v = (m @ getattr(self, f'cross_v_proj_{l}')).view(MAX_M, 8, 64).transpose(0, 1)
            ck_layers.append(k)
            cv_layers.append(v)
        mvalid = torch.cat([valid_cells, slot_valid], dim=0)
        cross_mask = (1.0 - mvalid) * -1e9
        return _stack_layers(ck_layers), _stack_layers(cv_layers), cross_mask


class MirageDecoderStep(nn.Module):
    """Exportable single-step autoregressive Mirage decoder."""

    def __init__(self, weights: dict[str, torch.Tensor], perm1: torch.Tensor, perm2: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer('perm1', perm1.clone().to(torch.int64))
        self.register_buffer('perm2', perm2.clone().to(torch.int64))

        lane_pre = torch.full((8, 4), -4.0, dtype=torch.float32)
        lane_post = torch.full((8, 4), -4.0, dtype=torch.float32)
        for l in range(8):
            lane_pre[l, l % 4] = 4.0
            lane_post[l, l % 4] = 0.0
        self.register_buffer('lane_pre', lane_pre)
        self.register_buffer('lane_post', lane_post)

        self.register_buffer('embed_T', weights['embedding/embedding'].T.contiguous().float())
        self.register_buffer('final_norm', weights['stack/final_norm/scale'].clone().float())

        for i in range(2):
            self.register_buffer(f'eng_{i}_key_proj', weights[f'engrams_{i}/key_proj/kernel'].clone().float())
            self.register_buffer(f'eng_{i}_value_proj', weights[f'engrams_{i}/value_proj/kernel'].clone().float())
            self.register_buffer(f'eng_{i}_taps', weights[f'engrams_{i}/taps'].clone().float())

        for l in range(8):
            self.register_buffer(f'mhc_a_pre_{l}', weights['stack/mhc_a_pre'][l : l + 1].clone().float())
            self.register_buffer(f'mhc_b_pre_{l}', weights['stack/mhc_b_pre'][l].clone().float())
            self.register_buffer(f'mhc_phi_pre_{l}', weights['stack/mhc_phi_pre'][l].clone().float())
            self.register_buffer(f'mhc_a_post_{l}', weights['stack/mhc_a_post'][l : l + 1].clone().float())
            self.register_buffer(f'mhc_b_post_{l}', weights['stack/mhc_b_post'][l].clone().float())
            self.register_buffer(f'mhc_phi_post_{l}', weights['stack/mhc_phi_post'][l].clone().float())
            self.register_buffer(f'mhc_a_res_{l}', weights['stack/mhc_a_res'][l : l + 1].clone().float())
            self.register_buffer(f'mhc_b_res_{l}', weights['stack/mhc_b_res'][l].clone().float())
            self.register_buffer(f'mhc_phi_res_{l}', weights['stack/mhc_phi_res'][l].clone().float())

            self.register_buffer(f'zc0_{l}', weights['stack/layers/block/ZCRMSNorm_0/scale'][l].clone().float())
            for p_name in ('q', 'k', 'v', 'gate', 'out'):
                self.register_buffer(
                    f'sa_{p_name}_proj_{l}',
                    weights[f'stack/layers/block/self_attn/{p_name}_proj/kernel'][l].clone().float(),
                )
            for p_name in ('q', 'k', 'v'):
                self.register_buffer(
                    f'sa_{p_name}_taps_{l}',
                    weights[f'stack/layers/block/self_attn/{p_name}_taps'][l].clone().float(),
                )
            self.register_buffer(f'sa_q_norm_{l}', weights['stack/layers/block/self_attn/q_norm/scale'][l].clone().float())
            self.register_buffer(f'sa_k_norm_{l}', weights['stack/layers/block/self_attn/k_norm/scale'][l].clone().float())
            self.register_buffer(f'post_attn_norm_{l}', weights['stack/layers/block/post_attn_norm/scale'][l].clone().float())
            self.register_buffer(f'attn_gate_{l}', weights['stack/layers/block/attn_gate'][l : l + 1].clone().float())

            self.register_buffer(f'cross_norm_{l}', weights['stack/layers/block/cross_norm/scale'][l].clone().float())
            self.register_buffer(f'ca_q_proj_{l}', weights['stack/layers/block/cross_attn/q_proj/kernel'][l].clone().float())
            self.register_buffer(f'ca_q_norm_{l}', weights['stack/layers/block/cross_attn/q_norm/scale'][l].clone().float())
            self.register_buffer(f'ca_gate_proj_{l}', weights['stack/layers/block/cross_attn/gate_proj/kernel'][l].clone().float())
            self.register_buffer(f'ca_out_proj_{l}', weights['stack/layers/block/cross_attn/out_proj/kernel'][l].clone().float())
            self.register_buffer(f'post_cross_norm_{l}', weights['stack/layers/block/post_cross_norm/scale'][l].clone().float())
            self.register_buffer(f'cross_gate_{l}', weights['stack/layers/block/cross_gate'][l : l + 1].clone().float())

            self.register_buffer(f'pre_hada_norm_{l}', weights['stack/layers/block/pre_hada_norm/scale'][l].clone().float())
            for h_name in ('d1', 'd2', 'b2', 'd3', 'd4', 'w1a', 'w1b', 'w2a', 'w2b', 'w3a', 'w3b', 'cond_v', 'cond_u'):
                self.register_buffer(
                    f'h_{h_name}_{l}',
                    weights[f'stack/layers/block/hadamard_mlp/{h_name}'][l].clone().float(),
                )

    def _kron(self, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.T @ z.view(16, 32) @ b).reshape(1, 512)

    def _hmlp(self, x: torch.Tensor, l: int) -> torch.Tensor:
        cond = 1.0 + F.softmax(x @ getattr(self, f'h_cond_v_{l}'), dim=-1) @ getattr(self, f'h_cond_u_{l}')
        z = self._kron(
            getattr(self, f'h_d1_{l}') * x,
            getattr(self, f'h_w1a_{l}'),
            getattr(self, f'h_w1b_{l}'),
        )[0, self.perm1].unsqueeze(0)
        z = self._kron(
            F.silu(getattr(self, f'h_d2_{l}') * cond * z + getattr(self, f'h_b2_{l}')),
            getattr(self, f'h_w2a_{l}'),
            getattr(self, f'h_w2b_{l}'),
        )[0, self.perm2].unsqueeze(0)
        z = self._kron(
            getattr(self, f'h_d3_{l}') * z,
            getattr(self, f'h_w3a_{l}'),
            getattr(self, f'h_w3b_{l}'),
        )
        return getattr(self, f'h_d4_{l}') * z

    def forward(
        self,
        token_emb: torch.Tensor,
        eng0_emb: torch.Tensor,
        eng1_emb: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        pos_onehot: torch.Tensor,
        self_mask: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        cross_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        raw_q_hist: torch.Tensor,
        raw_k_hist: torch.Tensor,
        raw_v_hist: torch.Tensor,
        eng_v_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_emb = math.sqrt(D_MODEL)
        X = (scale_emb * token_emb).unsqueeze(0).expand(4, D_MODEL)
        inv_sqrt_48 = 1.0 / math.sqrt(48.0)
        inv_sqrt_d = 1.0 / math.sqrt(D_MODEL)
        pos_keep = 1.0 - pos_onehot[None, :, None]
        pos_write = pos_onehot[None, :, None]
        e0_2d = eng0_emb.view(1, D_MODEL)
        e1_2d = eng1_emb.view(1, D_MODEL)

        new_k_list = []
        new_v_list = []
        new_rq_list = []
        new_rk_list = []
        new_rv_list = []
        new_eng_list = []

        for l in range(8):
            nx = rms_unit(X.reshape(1, -1))
            hpre = torch.sigmoid(
                getattr(self, f'mhc_a_pre_{l}') * (nx @ getattr(self, f'mhc_phi_pre_{l}')).squeeze(0)
                + getattr(self, f'mhc_b_pre_{l}')
                + self.lane_pre[l]
            )
            u = (hpre[:, None] * X).sum(dim=0, keepdim=True)  # [1, 512]
            x = u

            if l in ENGRAM_SITES:
                si = 0 if l == 3 else 1
                e_vec = e0_2d if si == 0 else e1_2d
                k_eng = e_vec @ getattr(self, f'eng_{si}_key_proj')  # [1, 512]
                vraw = (e_vec @ getattr(self, f'eng_{si}_value_proj')).squeeze(0)  # [512]
                etaps = getattr(self, f'eng_{si}_taps')
                ehist = eng_v_hist[si]
                v_eng = (
                    etaps[0] * vraw
                    + etaps[1] * ehist[2]
                    + etaps[2] * ehist[5]
                    + etaps[3] * ehist[8]
                ).view(1, D_MODEL)
                new_eng_list.append(torch.cat([vraw.unsqueeze(0), ehist[:8]], dim=0))
                alpha = torch.sigmoid((rms_unit(x) * rms_unit(k_eng)).sum() * inv_sqrt_d)
                x = x + alpha * v_eng

            skip = x
            xn = zc_norm(x, getattr(self, f'zc0_{l}'), dim=-1)  # [1, 512]

            rq = (xn @ getattr(self, f'sa_q_proj_{l}')).squeeze(0)
            rk = (xn @ getattr(self, f'sa_k_proj_{l}')).squeeze(0)
            rv = (xn @ getattr(self, f'sa_v_proj_{l}')).squeeze(0)

            qt = getattr(self, f'sa_q_taps_{l}')
            kt = getattr(self, f'sa_k_taps_{l}')
            vt = getattr(self, f'sa_v_taps_{l}')

            rqh = raw_q_hist[l]
            rkh = raw_k_hist[l]
            rvh = raw_v_hist[l]

            q_mix = (qt[0] * rq + qt[1] * rqh[0] + qt[2] * rqh[1]).view(8, 48)
            k_mix = (kt[0] * rk + kt[1] * rkh[0] + kt[2] * rkh[1]).view(2, 48)
            v_mix = (vt[0] * rv + vt[1] * rvh[0] + vt[2] * rvh[1]).view(2, 64)

            new_rq_list.append(torch.stack([rq, rqh[0]], dim=0))
            new_rk_list.append(torch.stack([rk, rkh[0]], dim=0))
            new_rv_list.append(torch.stack([rv, rvh[0]], dim=0))

            q = rope_half_step(zc_norm(q_mix, getattr(self, f'sa_q_norm_{l}'), dim=-1), rope_cos, rope_sin)
            k = rope_half_step(zc_norm(k_mix, getattr(self, f'sa_k_norm_{l}'), dim=-1), rope_cos, rope_sin)

            pk_new = past_k[l] * pos_keep + k[:, None, :] * pos_write
            pv_new = past_v[l] * pos_keep + v_mix[:, None, :] * pos_write
            new_k_list.append(pk_new)
            new_v_list.append(pv_new)

            sa_scores = (q.view(2, 4, 48) @ pk_new.transpose(-1, -2)) * inv_sqrt_48
            sa_att = F.softmax(sa_scores + self_mask[None, None, :], dim=-1)
            sa_o = (sa_att @ pv_new).reshape(1, 512)
            A = (sa_o * torch.sigmoid(xn @ getattr(self, f'sa_gate_proj_{l}'))) @ getattr(self, f'sa_out_proj_{l}')
            x = skip + torch.sigmoid(getattr(self, f'attn_gate_{l}')) * zc_norm(A, getattr(self, f'post_attn_norm_{l}'), dim=-1)

            skip = x
            xc = zc_norm(x, getattr(self, f'cross_norm_{l}'), dim=-1)
            qc = zc_norm((xc @ getattr(self, f'ca_q_proj_{l}')).view(8, 48), getattr(self, f'ca_q_norm_{l}'), dim=-1)
            sc = (qc[:, None, :] @ cross_k[l].transpose(-1, -2))[:, 0, :] * inv_sqrt_48
            attc = F.softmax(sc + cross_mask[None, :], dim=-1)
            oc = (attc[:, None, :] @ cross_v[l])[:, 0, :].reshape(1, 512)
            CA = (oc * torch.sigmoid(xc @ getattr(self, f'ca_gate_proj_{l}'))) @ getattr(self, f'ca_out_proj_{l}')
            x = skip + torch.sigmoid(getattr(self, f'cross_gate_{l}')) * zc_norm(CA, getattr(self, f'post_cross_norm_{l}'), dim=-1)

            x = x + self._hmlp(zc_norm(x, getattr(self, f'pre_hada_norm_{l}'), dim=-1), l)
            y = (x - u).squeeze(0)  # [512]

            hpost = 2.0 * torch.sigmoid(
                getattr(self, f'mhc_a_post_{l}') * (nx @ getattr(self, f'mhc_phi_post_{l}')).squeeze(0)
                + getattr(self, f'mhc_b_post_{l}')
                + self.lane_post[l]
            )
            hres = sinkhorn_4x4(
                getattr(self, f'mhc_a_res_{l}') * (nx @ getattr(self, f'mhc_phi_res_{l}')).view(4, 4)
                + getattr(self, f'mhc_b_res_{l}')
            )
            X = hres @ X + hpost[:, None] * y[None, :]

        h = zc_norm(X.mean(dim=0, keepdim=True), self.final_norm, dim=-1)
        logits = (h @ self.embed_T).squeeze(0)
        return (
            logits,
            _stack_layers(new_k_list),
            _stack_layers(new_v_list),
            _stack_layers(new_rq_list),
            _stack_layers(new_rk_list),
            _stack_layers(new_rv_list),
            _stack_layers(new_eng_list),
        )
