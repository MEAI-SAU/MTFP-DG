import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ContinuousTimeEmbedding(nn.Module):

    def __init__(self, d_time: int):
        super().__init__()
        self.d_time = d_time
        self.omega = nn.Parameter(torch.randn(d_time) * 0.01)
        self.alpha = nn.Parameter(torch.zeros(d_time))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*t.shape, self.d_time, device=t.device, dtype=t.dtype)
        out[..., 0] = self.omega[0] * t + self.alpha[0]
        if self.d_time > 1:
            out[..., 1:] = torch.sin(t.unsqueeze(-1) * self.omega[1:] + self.alpha[1:])
        return out


class MetaFilterMLP(nn.Module):

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TTCN(nn.Module):

    def __init__(self, d_v: int, d_time: int, d_out: int, d_hidden: int = 64):
        super().__init__()
        self.d_in = d_v + d_time
        self.d_out = d_out
        self.time_embed = ContinuousTimeEmbedding(d_time)
        self.meta_filters = nn.ModuleList([
            MetaFilterMLP(self.d_in, d_hidden) for _ in range(d_out)
        ])

    def forward(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(timestamps)
        z = torch.cat([t_emb, values], dim=-1)
        m = mask.unsqueeze(-1)
        outputs = []
        for fk in self.meta_filters:
            s = fk(z)
            s = s.masked_fill(~m.expand_as(s), float('-inf'))
            fw = torch.nan_to_num(F.softmax(s, dim=-2), nan=0.0)
            fw = fw.masked_fill(~m.expand_as(fw), 0.0)
            outputs.append((fw * z).sum(dim=-2).sum(dim=-1))
        return torch.stack(outputs, dim=-1)


class IFANLayer(nn.Module):

    def __init__(self, d_in: int, d_p: int, d_pbar: int):
        super().__init__()
        self.Wp = nn.Linear(d_in, d_p, bias=False)
        self.Up = nn.Linear(1, d_p, bias=False)
        self.Wp_bar = nn.Linear(d_in, d_pbar, bias=False)
        self.Vp_bar = nn.Linear(1, d_pbar, bias=False)
        self.Bp_bar = nn.Parameter(torch.zeros(d_pbar))
        self.d_out = 2 * d_p + d_pbar

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_u = t.unsqueeze(-1)
        u_p = self.Wp(x) + self.Up(t_u)
        g = F.gelu(self.Bp_bar + self.Wp_bar(x) + self.Vp_bar(t_u))
        return torch.cat([torch.cos(u_p), torch.sin(u_p), g], dim=-1)


class IFAN(nn.Module):

    def __init__(self, d_v: int, d_p: int, d_pbar: int, n_layers: int, d_out: int):
        super().__init__()
        self.layers = nn.ModuleList()
        d_in = d_v
        for _ in range(n_layers):
            layer = IFANLayer(d_in, d_p, d_pbar)
            self.layers.append(layer)
            d_in = layer.d_out
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(
        self,
        values: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = values
        for layer in self.layers:
            x = layer(x, timestamps)
        x = self.proj(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(dim=-2) / m.sum(dim=-2).clamp(min=1.0)


class DualDomainFusion(nn.Module):

    def __init__(self, d_time: int, d_freq: int, d_fused: int):
        super().__init__()
        self.proj = nn.Linear(d_time + d_freq, d_fused)

    def forward(self, h_time: torch.Tensor, h_freq: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([h_time, h_freq], dim=-1))


class MultiScaleIntegration(nn.Module):

    def __init__(self, n_scales: int, d_fused: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(n_scales * d_fused, d_out)

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        return F.gelu(self.proj(torch.cat(embeds, dim=-1)))


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = x.shape[1]
        return self.encoder(x + self.pos_emb[:, :P, :])


class LombScargleFeatures(nn.Module):

    def __init__(self, n_freqs: int, d_fourier: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.proj = nn.Linear(2 * n_freqs, d_fourier)

    def forward(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        device = timestamps.device
        freqs = torch.linspace(0.5 / self.n_freqs, 0.5, self.n_freqs, device=device)
        t = timestamps.unsqueeze(-1)
        f = freqs.view(*([1] * timestamps.dim()), -1)
        phase = 2.0 * math.pi * f * t
        m = mask.unsqueeze(-1).float()
        v = values.mean(dim=-1, keepdim=True)
        cnt = m.sum(dim=-2).clamp(min=1.0)
        cos_f = (torch.cos(phase) * v * m).sum(dim=-2) / cnt
        sin_f = (torch.sin(phase) * v * m).sum(dim=-2) / cnt
        return self.proj(torch.cat([cos_f, sin_f], dim=-1))


class DynamicGraphConstruction(nn.Module):

    def __init__(self, d_comb: int, d_g: int):
        super().__init__()
        self.W_src = nn.Linear(d_comb, d_g, bias=False)
        self.W_tgt = nn.Linear(d_comb, d_g, bias=False)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        E_src = self.W_src(C)
        E_tgt = self.W_tgt(C)
        A = F.relu(torch.matmul(E_src, E_tgt.transpose(-1, -2)))
        return F.softmax(A, dim=-1)


class GCNLayer(nn.Module):

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        A_hat = A + I
        d_inv_sqrt = A_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8).pow(-0.5)
        A_norm = d_inv_sqrt * A_hat * d_inv_sqrt.transpose(-1, -2)
        return F.gelu(self.W(torch.matmul(A_norm, x)))


class IFANPredictionLayer(nn.Module):

    def __init__(self, d_ctx: int, d_v: int, d_p: int, d_pbar: int):
        super().__init__()
        self.Wp = nn.Linear(d_ctx, d_p, bias=False)
        self.Up = nn.Linear(1, d_p, bias=False)
        self.Wp_bar = nn.Linear(d_ctx, d_pbar, bias=False)
        self.Vp_bar = nn.Linear(1, d_pbar, bias=False)
        self.Bp_bar = nn.Parameter(torch.zeros(d_pbar))
        self.out_proj = nn.Linear(2 * d_p + d_pbar, d_v)

    def forward(self, ctx: torch.Tensor, t_fut: torch.Tensor) -> torch.Tensor:
        B, N, D = ctx.shape
        if t_fut.dim() == 1:
            t_fut = t_fut.unsqueeze(0).expand(B, -1)
        H = t_fut.shape[1]
        ctx_exp = ctx.unsqueeze(2).expand(B, N, H, D)
        t_exp = t_fut.unsqueeze(1).unsqueeze(-1).expand(B, N, H, 1)
        u_p = self.Wp(ctx_exp) + self.Up(t_exp)
        g = F.gelu(self.Bp_bar + self.Wp_bar(ctx_exp) + self.Vp_bar(t_exp))
        feat = torch.cat([torch.cos(u_p), torch.sin(u_p), g], dim=-1)
        return self.out_proj(feat)


class MTFPDGBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_tf_layers: int,
        d_ff: int,
        d_fourier: int,
        d_g: int,
        n_lomb_freqs: int,
        P1: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = TemporalTransformer(
            d_model, n_heads, n_tf_layers, d_ff, P1 + 64, dropout
        )
        self.lomb = LombScargleFeatures(n_lomb_freqs, d_fourier)
        self.graph_ctor = DynamicGraphConstruction(d_model + d_fourier, d_g)
        self.gnn = GCNLayer(d_model, d_model)

    def forward(
        self,
        m_tokens: torch.Tensor,
        pat_t: torch.Tensor,
        pat_v: torch.Tensor,
        pat_m: torch.Tensor,
    ) -> torch.Tensor:
        B, N, P1, D = m_tokens.shape
        x = self.transformer(m_tokens.view(B * N, P1, D)).view(B, N, P1, D)
        lomb_feats = self.lomb(pat_t, pat_v, pat_m)
        C = torch.cat([x, lomb_feats], dim=-1).permute(0, 2, 1, 3)
        A = self.graph_ctor(C)
        out = self.gnn(x.permute(0, 2, 1, 3), A).permute(0, 2, 1, 3)
        return out


class MTFPDG(nn.Module):

    def __init__(
        self,
        n_vars: int,
        d_v: int = 1,
        patch_scales: Optional[List[float]] = None,
        d_time: int = 10,
        d_fused: int = 64,
        d_model: int = 64,
        n_heads: int = 1,
        n_tf_layers: int = 1,
        d_ff: int = 128,
        d_fourier: int = 16,
        d_g: int = 32,
        n_blocks: int = 1,
        ifan_layers: int = 2,
        ifan_dp: int = 16,
        ifan_dpbar: int = 16,
        pred_dp: int = 16,
        pred_dpbar: int = 16,
        T: float = 24.0,
        ttcn_d_hidden: int = 64,
        n_lomb_freqs: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_scales is None:
            patch_scales = [1.0, 2.0, 4.0, 8.0, 24.0]

        self.n_vars = n_vars
        self.d_v = d_v
        self.patch_scales = sorted(patch_scales)
        self.n_scales = len(self.patch_scales)
        self.T = T
        self.d_model = d_model

        self.n_patches = [max(1, int(T / s)) for s in self.patch_scales]
        self.P1 = self.n_patches[0]
        d_h = d_fused // 2

        self.ttcn_list = nn.ModuleList([
            TTCN(d_v, d_time, d_h, ttcn_d_hidden) for _ in self.patch_scales
        ])
        self.ifan_list = nn.ModuleList([
            IFAN(d_v, ifan_dp, ifan_dpbar, ifan_layers, d_h) for _ in self.patch_scales
        ])
        self.fusion_list = nn.ModuleList([
            DualDomainFusion(d_h, d_h, d_fused) for _ in self.patch_scales
        ])
        self.ms_integration = MultiScaleIntegration(self.n_scales, d_fused, d_model)
        self.blocks = nn.ModuleList([
            MTFPDGBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_tf_layers=n_tf_layers,
                d_ff=d_ff,
                d_fourier=d_fourier,
                d_g=d_g,
                n_lomb_freqs=n_lomb_freqs,
                P1=self.P1,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])
        self.pred_head = IFANPredictionLayer(d_model, d_v, pred_dp, pred_dpbar)

    def _extract_patches(
        self,
        obs_times: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        patch_scale: float,
        n_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, L = obs_times.shape
        D_v = obs_values.shape[-1]
        device = obs_times.device

        patch_idx = (obs_times / patch_scale).long().clamp(0, n_patches - 1)

        max_obs = 1
        for p in range(n_patches):
            in_p = obs_mask & (patch_idx == p)
            c = int(in_p.long().sum(dim=-1).max().item())
            if c > max_obs:
                max_obs = c

        p_times = obs_times.new_zeros(B, N, n_patches, max_obs)
        p_vals = obs_values.new_zeros(B, N, n_patches, max_obs, D_v)
        p_mask = obs_mask.new_zeros(B, N, n_patches, max_obs)

        for p in range(n_patches):
            t_start = p * patch_scale
            in_p = obs_mask & (patch_idx == p)
            for b in range(B):
                for n_i in range(N):
                    idx = in_p[b, n_i].nonzero(as_tuple=True)[0]
                    cnt = len(idx)
                    if cnt > 0:
                        p_times[b, n_i, p, :cnt] = (obs_times[b, n_i, idx] - t_start) / patch_scale
                        p_vals[b, n_i, p, :cnt] = obs_values[b, n_i, idx]
                        p_mask[b, n_i, p, :cnt] = True

        return p_times, p_vals, p_mask

    def _align_to_canonical(self, h: torch.Tensor, s_idx: int) -> torch.Tensor:
        B, N, Ps, D = h.shape
        if Ps == self.P1:
            return h
        L1 = self.patch_scales[0]
        Ls = self.patch_scales[s_idx]
        p_fine = torch.arange(self.P1, device=h.device)
        p_coarse = (p_fine.float() * (L1 / Ls)).long().clamp(0, Ps - 1)
        return h[:, :, p_coarse, :]

    def forward(
        self,
        obs_times: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        B = obs_times.shape[0]
        scale_embeds = []
        fine_patches: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        for s_idx, (sc, np_s, ttcn, ifan_m, fus) in enumerate(zip(
            self.patch_scales, self.n_patches,
            self.ttcn_list, self.ifan_list, self.fusion_list,
        )):
            pt, pv, pm = self._extract_patches(obs_times, obs_values, obs_mask, sc, np_s)
            if s_idx == 0:
                fine_patches = (pt, pv, pm)
            h_t = ttcn(pt, pv, pm)
            h_f = ifan_m(pv, pt, pm)
            h_fused = fus(h_t, h_f)
            scale_embeds.append(self._align_to_canonical(h_fused, s_idx))

        m_tokens = self.ms_integration(scale_embeds)

        for block in self.blocks:
            m_tokens = block(m_tokens, *fine_patches)

        ctx = m_tokens[:, :, -1, :]

        if query_times.dim() == 1:
            query_times = query_times.unsqueeze(0).expand(B, -1)
        q_norm = query_times / self.T

        return self.pred_head(ctx, q_norm)

    def compute_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if target_mask is not None:
            m = target_mask.unsqueeze(-1).float()
            diff = (preds - targets) * m
            n = m.sum().clamp(min=1.0)
            mse = (diff ** 2).sum() / n
            mae = diff.abs().sum() / n
        else:
            mse = F.mse_loss(preds, targets)
            mae = F.l1_loss(preds, targets)
        rmse = mse.sqrt()
        mre = (mae / (targets.abs().mean().clamp(min=1e-8))).item()
        return {
            "MSE": mse.item(),
            "MAE": mae.item(),
            "RMSE": rmse.item(),
            "MRE": mre,
        }


class MedicalTimeAwareLoss(nn.Module):

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        query_times: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        if query_times.dim() == 1:
            query_times = query_times.unsqueeze(0)
        w = (1.0 + self.alpha * (query_times - T).clamp(min=0.0))
        w = w.unsqueeze(1).unsqueeze(-1)
        return (w * (preds - targets).pow(2)).mean()


def build_model(config: dict) -> MTFPDG:
    return MTFPDG(
        n_vars=config.get("n_vars", 41),
        d_v=config.get("d_v", 1),
        patch_scales=config.get("patch_scales", [1.0, 2.0, 4.0, 8.0, 24.0]),
        d_time=config.get("d_time", 10),
        d_fused=config.get("d_fused", 64),
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 1),
        n_tf_layers=config.get("n_tf_layers", 1),
        d_ff=config.get("d_ff", 128),
        d_fourier=config.get("d_fourier", 16),
        d_g=config.get("d_g", 32),
        n_blocks=config.get("n_blocks", 1),
        ifan_layers=config.get("ifan_layers", 2),
        ifan_dp=config.get("ifan_dp", 16),
        ifan_dpbar=config.get("ifan_dpbar", 16),
        pred_dp=config.get("pred_dp", 16),
        pred_dpbar=config.get("pred_dpbar", 16),
        T=config.get("T", 24.0),
        ttcn_d_hidden=config.get("ttcn_d_hidden", 64),
        n_lomb_freqs=config.get("n_lomb_freqs", 8),
        dropout=config.get("dropout", 0.1),
    )


def train_step(
    model: MTFPDG,
    optimizer: torch.optim.Optimizer,
    criterion: MedicalTimeAwareLoss,
    obs_times: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    query_times: torch.Tensor,
    targets: torch.Tensor,
    T: float,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad()
    preds = model(obs_times, obs_values, obs_mask, query_times)
    loss = criterion(preds, targets, query_times, T)
    loss.backward()
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_step(
    model: MTFPDG,
    obs_times: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    query_times: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    model.eval()
    preds = model(obs_times, obs_values, obs_mask, query_times)
    return model.compute_metrics(preds, targets)


if __name__ == "__main__":
    torch.manual_seed(42)

    B, N, L = 2, 10, 50
    D_v = 1
    H_prime = 24
    T = 24.0

    obs_times = torch.rand(B, N, L) * T
    obs_times, _ = obs_times.sort(dim=-1)
    obs_values = torch.randn(B, N, L, D_v)
    obs_mask = torch.ones(B, N, L, dtype=torch.bool)
    obs_mask[:, :, 40:] = False
    query_times = torch.linspace(24.5, 48.0, H_prime).unsqueeze(0).expand(B, -1)

    config = {
        "n_vars": N,
        "d_v": D_v,
        "patch_scales": [1.0, 2.0, 4.0, 8.0, 24.0],
        "d_time": 10,
        "d_fused": 32,
        "d_model": 32,
        "n_heads": 1,
        "n_tf_layers": 1,
        "d_ff": 64,
        "d_fourier": 8,
        "d_g": 16,
        "n_blocks": 1,
        "ifan_layers": 1,
        "ifan_dp": 8,
        "ifan_dpbar": 8,
        "pred_dp": 8,
        "pred_dpbar": 8,
        "T": T,
        "ttcn_d_hidden": 32,
        "n_lomb_freqs": 4,
        "dropout": 0.0,
    }

    model = build_model(config)
    criterion = MedicalTimeAwareLoss(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    preds = model(obs_times, obs_values, obs_mask, query_times)
    print(f"Predictions shape: {preds.shape}")

    targets = torch.randn_like(preds)
    loss = criterion(preds, targets, query_times, T)
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass: OK")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    metrics = model.compute_metrics(preds.detach(), targets)
    print(f"Metrics: {metrics}")
