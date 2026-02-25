
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import torch.nn.functional as F
from typing import Tuple
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import warnings

# ---- Global perf flags ----
torch.backends.cudnn.benchmark = True  # usually helps for fixed-size inputs on GPU

# helper: keep CPU tensors by default in dataset conversion, then move to GPU in training loop
def df_to_bin_tensors(df: pd.DataFrame, max_particles: int = 200) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Liefert pT, eta, phi Tensors (B, N) mit -1 als Padding, auf CPU (nicht auf device).
    Effizientere Implementation: nur vorhandene Spalten werden genutzt.
    """
    pattern = re.compile(r'^(pT|eta|phi)_bin_(\d+)$')
    present_idxs = set()
    for col in df.columns:
        m = pattern.match(col)
        if m:
            present_idxs.add(int(m.group(2)))
    max_present = max(present_idxs) if present_idxs else -1

    B = len(df)
    N = max_particles

    pT_arr = np.full((B, N), -1, dtype=np.int64)
    eta_arr = np.full((B, N), -1, dtype=np.int64)
    phi_arr = np.full((B, N), -1, dtype=np.int64)

    for j in sorted(present_idxs):
        if j >= N:
            break
        pt_col = f"pT_bin_{j}"
        if pt_col in df.columns:
            pT_arr[:, j] = df[pt_col].fillna(-1).astype(np.int64).values
        eta_col = f"eta_bin_{j}"
        if eta_col in df.columns:
            eta_arr[:, j] = df[eta_col].fillna(-1).astype(np.int64).values
        phi_col = f"phi_bin_{j}"
        if phi_col in df.columns:
            phi_arr[:, j] = df[phi_col].fillna(-1).astype(np.int64).values

    if max_present >= 0:
        available = [f"pT_bin_{j}" for j in range(min(max_present + 1, N)) if f"pT_bin_{j}" in df.columns]
        if available:
            stacked = np.stack([df[c].fillna(-1).astype(np.int64).values for c in available], axis=1)  # (B, M)
            counts = np.sum(stacked != -1, axis=1).astype(np.int64)
        else:
            counts = np.zeros(B, dtype=np.int64)
    else:
        counts = np.zeros(B, dtype=np.int64)

    counts = np.minimum(counts, N)

    pT_t = torch.from_numpy(pT_arr).long()   # CPU tensor
    eta_t = torch.from_numpy(eta_arr).long()
    phi_t = torch.from_numpy(phi_arr).long()
    counts_t = torch.from_numpy(counts).long()

    return pT_t, eta_t, phi_t, counts_t


class ParticleEmbedder(nn.Module):
    def __init__(self,
                 emb_dim: int = 256,
                 pT_max_value: int = 40,
                 eta_max_value: int = 30,
                 phi_max_value: int = 30,
                 max_particles: int = 100,
                 use_position_embedding: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_particles = max_particles
        self.S = 1 + max_particles + 1  # START + N + STOP

        self.pT_emb = nn.Embedding(num_embeddings=pT_max_value + 2, embedding_dim=emb_dim, padding_idx=0)
        self.eta_emb = nn.Embedding(num_embeddings=eta_max_value + 2, embedding_dim=emb_dim, padding_idx=0)
        self.phi_emb = nn.Embedding(num_embeddings=phi_max_value + 2, embedding_dim=emb_dim, padding_idx=0)

        self.use_position_embedding = use_position_embedding
        if use_position_embedding:
            self.pos_emb = nn.Embedding(max_particles + 1, emb_dim)
            # precreate pos index buffer 1..N to avoid repeated arange allocations
            pos_idx = torch.arange(1, max_particles + 1).unsqueeze(0)  # (1, N)
            self.register_buffer("pos_idx_buffer", pos_idx, persistent=False)
        else:
            self.pos_emb = None
            self.register_buffer("pos_idx_buffer", torch.empty(0), persistent=False)

        self.start_token = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.stop_token = nn.Parameter(torch.randn(emb_dim) * 0.02)

        self.layernorm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, pT_bins: torch.LongTensor, eta_bins: torch.LongTensor, phi_bins: torch.LongTensor, counts: torch.LongTensor) -> torch.FloatTensor:
        B, N = pT_bins.shape
        assert N == self.max_particles
        device = pT_bins.device
        D = self.emb_dim
        S = self.S

        pT_idx = torch.clamp(pT_bins + 1, min=0, max=self.pT_emb.num_embeddings - 1)
        eta_idx = torch.clamp(eta_bins + 1, min=0, max=self.eta_emb.num_embeddings - 1)
        phi_idx = torch.clamp(phi_bins + 1, min=0, max=self.phi_emb.num_embeddings - 1)

        e_pT = self.pT_emb(pT_idx)
        e_eta = self.eta_emb(eta_idx)
        e_phi = self.phi_emb(phi_idx)
        e_particles = e_pT + e_eta + e_phi

        if self.use_position_embedding:
            pos_idx = self.pos_idx_buffer.to(device).expand(B, N)  # (B,N)
            e_particles = e_particles + self.pos_emb(pos_idx)

        seq = torch.zeros((B, S, D), dtype=e_particles.dtype, device=device)
        seq[:, 0, :] = self.start_token.view(1, D)

        arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        counts_exp = counts.unsqueeze(1).expand(B, N)
        shift = (arange >= counts_exp).long()
        dest = 1 + arange + shift  # (B, N)

        dest_idx = dest.unsqueeze(-1).expand(B, N, D)
        seq = seq.scatter(dim=1, index=dest_idx, src=e_particles)

        stop_needed = (counts < N)
        if stop_needed.any():
            stop_pos = (1 + counts).to(torch.long)
            batch_idx = torch.arange(B, device=device)
            seq[batch_idx[stop_needed], stop_pos[stop_needed], :] = self.stop_token.view(1, D)

        seq = self.layernorm(seq)
        seq = self.dropout(seq)
        return seq

    def make_masks(self, pT_bins: torch.LongTensor, counts: torch.LongTensor) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        B, N = pT_bins.shape
        device = pT_bins.device
        S = self.S

        orig_pad = (pT_bins == -1)
        arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        counts_exp = counts.unsqueeze(1).expand(B, N)
        shift = (arange >= counts_exp).long()
        dest = 1 + arange + shift

        src_key_padding_mask = torch.ones((B, S), dtype=torch.bool, device=device)
        src_key_padding_mask[:, 0] = False  # START unmasked

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        real_mask = ~orig_pad
        if real_mask.any():
            src_key_padding_mask[batch_idx[real_mask], dest[real_mask]] = False

        stop_needed = (counts < N)
        if stop_needed.any():
            stop_pos = (1 + counts)
            batch_idx_simple = torch.arange(B, device=device)
            src_key_padding_mask[batch_idx_simple[stop_needed], stop_pos[stop_needed]] = False

        causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device), diagonal=0)
        causal_mask[:, 0] = False
        causal_mask[0, 0] = False

        return src_key_padding_mask, causal_mask


class ParticleTransformer(nn.Module):
    def __init__(self,
                 num_pT_embeddings: int = 42,
                 num_eta_embeddings: int = 32,
                 num_phi_embeddings: int = 32,
                 emb_dim: int = 256,
                 n_layers: int = 8,
                 n_heads: int = 4,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 pad_token: int = 0):
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_token = pad_token
        if dim_feedforward is None:
            dim_feedforward = emb_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(emb_dim))

        self.final_norm = nn.LayerNorm(emb_dim)
        self.final_dropout = nn.Dropout(dropout)

        self.type_head = nn.Linear(emb_dim, 3)
        self.head_pT  = nn.Linear(emb_dim, num_pT_embeddings)
        self.head_eta = nn.Linear(emb_dim, num_eta_embeddings)
        self.head_phi = nn.Linear(emb_dim, num_phi_embeddings)

    def forward(self, seq_emb: torch.FloatTensor, src_key_padding_mask: torch.BoolTensor, causal_mask: torch.BoolTensor):
        B, S, D = seq_emb.shape
        assert D == self.emb_dim

        attn_mask = None
        if causal_mask is not None:
            attn_mask = causal_mask
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            assert attn_mask.shape == (S, S)

        enc = self.encoder(seq_emb, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        enc = self.final_norm(enc)
        enc = self.final_dropout(enc)

        logits_type = self.type_head(enc)
        logits_pT  = self.head_pT(enc)
        logits_eta = self.head_eta(enc)
        logits_phi = self.head_phi(enc)

        return logits_type, logits_pT, logits_eta, logits_phi


def prepare_batch_for_mode(pT_full: torch.LongTensor,
                           eta_full: torch.LongTensor,
                           phi_full: torch.LongTensor,
                           counts_full: torch.LongTensor,
                           mode: str = "train",
                           train_max_real: int = 50,
                           max_particles: int = 100) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    assert mode in ("train", "test")
    B, N = pT_full.shape
    assert N == max_particles
    device = pT_full.device

    if mode == "test":
        return pT_full.clone(), eta_full.clone(), phi_full.clone(), counts_full.clone()

    tm = int(train_max_real)
    assert tm <= max_particles

    pT_out = torch.full((B, max_particles), -1, dtype=pT_full.dtype, device=device)
    eta_out = torch.full((B, max_particles), -1, dtype=eta_full.dtype, device=device)
    phi_out = torch.full((B, max_particles), -1, dtype=phi_full.dtype, device=device)

    pT_out[:, :tm] = pT_full[:, :tm]
    eta_out[:, :tm] = eta_full[:, :tm]
    phi_out[:, :tm] = phi_full[:, :tm]

    counts_out = torch.clamp(counts_full, max=tm)

    return pT_out, eta_out, phi_out, counts_out


def build_targets_from_bins(pT_bins, eta_bins, phi_bins, counts,
                            num_pT_emb, num_eta_emb, num_phi_emb):
    B, N = pT_bins.shape
    pT_idx = torch.clamp(pT_bins + 1, min=0, max=num_pT_emb - 1)
    eta_idx = torch.clamp(eta_bins + 1, min=0, max=num_eta_emb - 1)
    phi_idx = torch.clamp(phi_bins + 1, min=0, max=num_phi_emb - 1)

    device = pT_bins.device
    arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    counts_exp = counts.unsqueeze(1).expand(B, N)
    shift = (arange >= counts_exp).long()
    dest = 1 + arange + shift

    S = 1 + N + 1
    type_tgt = torch.full((B, S), -100, dtype=torch.long, device=device)
    pT_tgt  = torch.full((B, S), -100, dtype=torch.long, device=device)
    eta_tgt = torch.full((B, S), -100, dtype=torch.long, device=device)
    phi_tgt = torch.full((B, S), -100, dtype=torch.long, device=device)

    type_tgt[:, 0] = 0

    orig_real = (pT_bins != -1)
    if orig_real.any():
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        real_batch_idx = batch_idx[orig_real]
        real_dest_idx = dest[orig_real]
        type_tgt[real_batch_idx, real_dest_idx] = 1

        pT_tgt[real_batch_idx, real_dest_idx] = pT_idx[orig_real]
        eta_tgt[real_batch_idx, real_dest_idx] = eta_idx[orig_real]
        phi_tgt[real_batch_idx, real_dest_idx] = phi_idx[orig_real]

    stop_needed = (counts < N)
    if stop_needed.any():
        stop_pos = (1 + counts).long()
        batch_idx_simple = torch.arange(B, device=device)
        type_tgt[batch_idx_simple[stop_needed], stop_pos[stop_needed]] = 2

    orig_pad = (pT_bins == -1)
    src_key_padding_mask = torch.ones((B, S), dtype=torch.bool, device=device)
    src_key_padding_mask[:, 0] = False
    if orig_real.any():
        src_key_padding_mask[real_batch_idx, real_dest_idx] = False
    if stop_needed.any():
        src_key_padding_mask[batch_idx_simple[stop_needed], stop_pos[stop_needed]] = False

    causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device), diagonal=0)
    return type_tgt, pT_tgt, eta_tgt, phi_tgt, src_key_padding_mask, causal_mask


def tensor_has_nan_or_inf(t: torch.Tensor) -> bool:
    if t is None:
        return False
    return bool(torch.isnan(t).any().item()) or bool(torch.isinf(t).any().item())


def model_has_nan_or_inf(model):
    bad = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        if tensor_has_nan_or_inf(p.detach()):
            bad.append(("param", name))
    for name, buf in model.named_buffers():
        if tensor_has_nan_or_inf(buf.detach()):
            bad.append(("buffer", name))
    return bad


def grads_have_nan_or_inf(model):
    bad = []
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        if tensor_has_nan_or_inf(g):
            try:
                m = float(g.abs().max().cpu().item())
            except Exception:
                m = None
            bad.append((name, m))
    return bad


def print_nan_diagnostics(tag, embedder, transformer, extra_msg=""):
    print("=== NaN Diagnostic ===", tag, extra_msg)
    bad_e = model_has_nan_or_inf(embedder)
    bad_t = model_has_nan_or_inf(transformer)
    if bad_e:
        print(" Embedder param/buffer NaNs:", bad_e[:20])
    else:
        print(" Embedder: keine NaNs in Parametern/Buffers")
    if bad_t:
        print(" Transformer param/buffer NaNs:", bad_t[:20])
    else:
        print(" Transformer: keine NaNs in Parametern/Buffers")
    print("======================")


def train_and_evaluate(embedder, transformer,
                       pT_train, eta_train, phi_train, counts_train,
                       pT_val, eta_val, phi_val, counts_val,
                       epochs=50,
                       warmup_epochs=5,
                       batch_size=64,
                       lr=5e-5,
                       weight_decay=1e-4,
                       train_max_real=50,
                       device=None,
                       max_grad_norm=1.0,
                       enable_detect_anomaly=False,
                       max_batches_to_check_train=None,
                       save_dir=None,
                       save_best=True,
                       save_every_n_epochs=None,
                       model_name_prefix="best_model",
                       use_amp=False,
                       num_workers=4,
                       pin_memory=True,
                       persistent_workers=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder.to(device)
    transformer.to(device)

    dataset_train = TensorDataset(pT_train, eta_train, phi_train, counts_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory and device.type == "cuda",
                              persistent_workers=(persistent_workers if num_workers > 0 else False))

    dataset_val = TensorDataset(pT_val, eta_val, phi_val, counts_val)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=max(0, num_workers//2), pin_memory=pin_memory and device.type == "cuda",
                            persistent_workers=False)

    loss_type_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_pT_fn  = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    loss_eta_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    loss_phi_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    optimizer = optim.Adam(list(embedder.parameters()) + list(transformer.parameters()), lr=lr, weight_decay=weight_decay, eps=1e-8)

    # Scheduler: ensure T_max >= 1
    cosine_T = max(1, epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    history = {"train": [], "val": []}

    if enable_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    scaler = torch.amp.GradScaler(device_type="cuda") if (use_amp and device.type == "cuda") else torch.amp.GradScaler(enabled=False)

    batch_global_count = 0
    best_val_total = math.inf
    best_path = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    try:
        for epoch in range(1, epochs + 1):
            embedder.train()
            transformer.train()

            tr_type_sum = tr_type_count = 0.0
            tr_pT_sum = tr_pT_count = 0.0
            tr_eta_sum = tr_eta_count = 0.0
            tr_phi_sum = tr_phi_count = 0.0

            pbar = tqdm(loader_train, desc=f"Epoch {epoch}/{epochs} Train", leave=False)
            for batch in pbar:
                # DataLoader yields CPU tensors (pin_memory True), transfer non_blocking
                pT_b, eta_b, phi_b, counts_b = [t.to(device, non_blocking=True) for t in batch]
                batch_global_count += 1

                pT_in, eta_in, phi_in, counts_in = prepare_batch_for_mode(
                    pT_b, eta_b, phi_b, counts_b,
                    mode="train", train_max_real=train_max_real, max_particles=pT_b.shape[1]
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                    seq = embedder(pT_in, eta_in, phi_in, counts_in)
                    src_key_padding_mask, causal_mask = embedder.make_masks(pT_in, counts_in)
                    logits_type, logits_pT, logits_eta, logits_phi = transformer(seq, src_key_padding_mask, causal_mask)

                    B, S, _ = logits_type.shape

                    type_tgt, pT_tgt, eta_tgt, phi_tgt, _, _ = build_targets_from_bins(
                        pT_in, eta_in, phi_in, counts_in,
                        transformer.head_pT.out_features, transformer.head_eta.out_features, transformer.head_phi.out_features
                    )

                    mask_unmasked = ~(_ if False else torch.zeros_like(src_key_padding_mask))  # unused, replaced below

                    mask_unmasked = ~src_key_padding_mask
                    valid_idx = mask_unmasked.view(-1)
                    type_count = int(valid_idx.sum().item())
                    if type_count > 0:
                        logits_type_flat = logits_type.view(B*S, -1)[valid_idx]
                        type_tgt_flat = type_tgt.view(B*S)[valid_idx]
                        L_type_sum = loss_type_fn(logits_type_flat, type_tgt_flat)
                    else:
                        L_type_sum = torch.tensor(0.0, device=device)

                    logits_pT_flat = logits_pT.view(B*S, -1)
                    logits_eta_flat = logits_eta.view(B*S, -1)
                    logits_phi_flat = logits_phi.view(B*S, -1)
                    pT_targets_flat = pT_tgt.view(B*S)
                    eta_targets_flat = eta_tgt.view(B*S)
                    phi_targets_flat = phi_tgt.view(B*S)

                    pT_count = int((pT_targets_flat != -100).sum().item())
                    eta_count = int((eta_targets_flat != -100).sum().item())
                    phi_count = int((phi_targets_flat != -100).sum().item())

                    L_pT_sum  = loss_pT_fn(logits_pT_flat, pT_targets_flat) if pT_count > 0 else torch.tensor(0.0, device=device)
                    L_eta_sum = loss_eta_fn(logits_eta_flat, eta_targets_flat) if eta_count > 0 else torch.tensor(0.0, device=device)
                    L_phi_sum = loss_phi_fn(logits_phi_flat, phi_targets_flat) if phi_count > 0 else torch.tensor(0.0, device=device)

                    L_type_mean = (L_type_sum / type_count) if type_count > 0 else torch.tensor(0.0, device=device)
                    L_pT_mean   = (L_pT_sum / pT_count) if pT_count > 0 else torch.tensor(0.0, device=device)
                    L_eta_mean  = (L_eta_sum / eta_count) if eta_count > 0 else torch.tensor(0.0, device=device)
                    L_phi_mean  = (L_phi_sum / phi_count) if phi_count > 0 else torch.tensor(0.0, device=device)

                    batch_loss = L_type_mean + L_pT_mean + L_eta_mean + L_phi_mean

                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    print("FEHLER: batch_loss ist NaN/Inf vor backward. Abbruch und Diagnose.")
                    print_nan_diagnostics(f"epoch{epoch}_batch{batch_global_count}", embedder, transformer,
                                          extra_msg=f"type_count={type_count}, pT_count={pT_count}, eta_count={eta_count}, phi_count={phi_count}")
                    raise RuntimeError("batch_loss NaN/Inf vor backward")

                if use_amp and device.type == "cuda":
                    scaler.scale(batch_loss).backward()
                    scaler.unscale_(optimizer)
                    bad_grads = grads_have_nan_or_inf(embedder) + grads_have_nan_or_inf(transformer)
                    if bad_grads:
                        print("FEHLER: NaNs/Inf in Gradienten nach backward:", bad_grads[:10])
                        print_nan_diagnostics(f"after_backward epoch{epoch}_batch{batch_global_count}", embedder, transformer)
                        raise RuntimeError("NaNs in Gradients detected")
                    torch.nn.utils.clip_grad_norm_(list(embedder.parameters()) + list(transformer.parameters()), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    bad_grads = grads_have_nan_or_inf(embedder) + grads_have_nan_or_inf(transformer)
                    if bad_grads:
                        print("FEHLER: NaNs/Inf in Gradienten nach backward:", bad_grads[:10])
                        print_nan_diagnostics(f"after_backward epoch{epoch}_batch{batch_global_count}", embedder, transformer)
                        raise RuntimeError("NaNs in Gradients detected")
                    torch.nn.utils.clip_grad_norm_(list(embedder.parameters()) + list(transformer.parameters()), max_grad_norm)
                    optimizer.step()

                tr_type_sum += float(L_type_sum.item())
                tr_type_count += type_count
                tr_pT_sum += float(L_pT_sum.item())
                tr_pT_count += pT_count
                tr_eta_sum += float(L_eta_sum.item())
                tr_eta_count += eta_count
                tr_phi_sum += float(L_phi_sum.item())
                tr_phi_count += phi_count

                if max_batches_to_check_train is not None and batch_global_count >= int(max_batches_to_check_train):
                    pbar.close()
                    print(f"Debug-Abbruch nach {batch_global_count} Trainingsbatches (max_batches_to_check_train).")
                    break

            train_type_avg = tr_type_sum / tr_type_count if tr_type_count > 0 else float("nan")
            train_pT_avg  = tr_pT_sum  / tr_pT_count  if tr_pT_count  > 0 else float("nan")
            train_eta_avg = tr_eta_sum / tr_eta_count if tr_eta_count > 0 else float("nan")
            train_phi_avg = tr_phi_sum / tr_phi_count if tr_phi_count > 0 else float("nan")
            train_total = sum(v for v in [train_type_avg, train_pT_avg, train_eta_avg, train_phi_avg] if not (isinstance(v, float) and np.isnan(v)))
            history["train"].append({"type": train_type_avg, "pT": train_pT_avg, "eta": train_eta_avg, "phi": train_phi_avg, "total": train_total})

            # Validation
            embedder.eval()
            transformer.eval()

            val_type_sum = val_type_count = 0.0
            val_pT_sum = val_pT_count = 0.0
            val_eta_sum = val_eta_count = 0.0
            val_phi_sum = val_phi_count = 0.0

            with torch.no_grad():
                for batch in tqdm(loader_val, desc=f"Epoch {epoch}/{epochs} Val", leave=False):
                    pT_b, eta_b, phi_b, counts_b = [t.to(device, non_blocking=True) for t in batch]
                    pT_in, eta_in, phi_in, counts_in = prepare_batch_for_mode(
                        pT_b, eta_b, phi_b, counts_b,
                        mode="test", train_max_real=train_max_real, max_particles=pT_b.shape[1]
                    )

                    seq = embedder(pT_in, eta_in, phi_in, counts_in)
                    src_key_padding_mask, causal_mask = embedder.make_masks(pT_in, counts_in)
                    logits_type, logits_pT, logits_eta, logits_phi = transformer(seq, src_key_padding_mask, causal_mask)

                    if tensor_has_nan_or_inf(logits_type) or tensor_has_nan_or_inf(logits_pT) or tensor_has_nan_or_inf(logits_eta) or tensor_has_nan_or_inf(logits_phi):
                        print("FEHLER: NaN/Inf in Logits während Validation erkannt. Diagnose:")
                        print_nan_diagnostics(f"val_epoch{epoch}", embedder, transformer, extra_msg="Logits NaN in validation")
                        raise RuntimeError("NaN/Inf in validation logits -- stop and inspect training run")

                    type_tgt, pT_tgt, eta_tgt, phi_tgt, _, _ = build_targets_from_bins(
                        pT_in, eta_in, phi_in, counts_in,
                        transformer.head_pT.out_features, transformer.head_eta.out_features, transformer.head_phi.out_features
                    )

                    B, S, _ = logits_type.shape
                    mask_unmasked = ~src_key_padding_mask
                    valid_idx = mask_unmasked.view(-1)
                    type_count = int(valid_idx.sum().item())

                    if type_count > 0:
                        logits_type_flat = logits_type.view(B*S, -1)[valid_idx]
                        type_tgt_flat = type_tgt.view(B*S)[valid_idx]
                        L_type_sum = loss_type_fn(logits_type_flat, type_tgt_flat)
                    else:
                        L_type_sum = torch.tensor(0.0, device=device)

                    logits_pT_flat = logits_pT.view(B*S, -1)
                    logits_eta_flat = logits_eta.view(B*S, -1)
                    logits_phi_flat = logits_phi.view(B*S, -1)
                    pT_targets_flat = pT_tgt.view(B*S)
                    eta_targets_flat = eta_tgt.view(B*S)
                    phi_targets_flat = phi_tgt.view(B*S)

                    pT_count = int((pT_targets_flat != -100).sum().item())
                    eta_count = int((eta_targets_flat != -100).sum().item())
                    phi_count = int((phi_targets_flat != -100).sum().item())

                    L_pT_sum = loss_pT_fn(logits_pT_flat, pT_targets_flat) if pT_count > 0 else torch.tensor(0.0, device=device)
                    L_eta_sum = loss_eta_fn(logits_eta_flat, eta_targets_flat) if eta_count > 0 else torch.tensor(0.0, device=device)
                    L_phi_sum = loss_phi_fn(logits_phi_flat, phi_targets_flat) if phi_count > 0 else torch.tensor(0.0, device=device)

                    val_type_sum += float(L_type_sum.item()); val_type_count += type_count
                    val_pT_sum += float(L_pT_sum.item());   val_pT_count += pT_count
                    val_eta_sum += float(L_eta_sum.item()); val_eta_count += eta_count
                    val_phi_sum += float(L_phi_sum.item()); val_phi_count += phi_count

            val_type_avg = val_type_sum / val_type_count if val_type_count > 0 else float("nan")
            val_pT_avg  = val_pT_sum  / val_pT_count  if val_pT_count  > 0 else float("nan")
            val_eta_avg = val_eta_sum / val_eta_count if val_eta_count > 0 else float("nan")
            val_phi_avg = val_phi_sum / val_phi_count if val_phi_count > 0 else float("nan")
            val_total = sum(v for v in [val_type_avg, val_pT_avg, val_eta_avg, val_phi_avg] if not (isinstance(v, float) and np.isnan(v)))
            history["val"].append({"type": val_type_avg, "pT": val_pT_avg, "eta": val_eta_avg, "phi": val_phi_avg, "total": val_total})

            scheduler.step()

            def fmt(x): return f"{x:.6f}" if not (isinstance(x, float) and np.isnan(x)) else "nan"
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train total={fmt(train_total)} type={fmt(train_type_avg)} pT={fmt(train_pT_avg)} eta={fmt(train_eta_avg)} phi={fmt(train_phi_avg)} || "
                f"Val total={fmt(val_total)} type={fmt(val_type_avg)} pT={fmt(val_pT_avg)} eta={fmt(val_eta_avg)} phi={fmt(val_phi_avg)}"
            )

            # --- Checkpointing: Save best model based on val_total (niedrigster) ---
            is_better = (not math.isnan(val_total)) and (val_total < best_val_total)
            if save_dir is not None and (save_best and is_better):
                best_val_total = val_total
                fname = f"{model_name_prefix}.pt"
                path = os.path.join(save_dir, fname)
                torch.save({
                    "epoch": epoch,
                    "val_total": val_total,
                    "embedder_state_dict": embedder.state_dict(),
                    "transformer_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history
                }, path)
                torch.save({
                    "embedder": embedder.state_dict(),
                    "transformer": transformer.state_dict(),
                }, os.path.join(save_dir, f"{model_name_prefix}_weights.pt"))

            if save_dir is not None and save_every_n_epochs is not None and epoch % save_every_n_epochs == 0:
                fname = f"checkpoint_epoch{epoch:03d}.pt"
                path = os.path.join(save_dir, fname)
                torch.save({
                    "epoch": epoch,
                    "val_total": val_total,
                    "embedder_state_dict": embedder.state_dict(),
                    "transformer_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history
                }, path)

            if max_batches_to_check_train is not None and batch_global_count >= int(max_batches_to_check_train):
                print("Debug: Abbruch nach Erreichen von max_batches_to_check_train")
                break

    finally:
        if enable_detect_anomaly:
            torch.autograd.set_detect_anomaly(False)

    train_df = pd.DataFrame(history["train"])
    val_df = pd.DataFrame(history["val"])
    history_df = pd.concat([train_df.add_prefix("train_"), val_df.add_prefix("val_")], axis=1)

    return history_df, path


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*epoch parameter in `scheduler.step\\(\\)` was not necessary.*")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    #device = torch.device("cuda")

    max_particles = 100
    train_max_real = 50

    train_data = pd.read_hdf("/home/ew640340/Ph.D./Top_Quark_Tagging_Ref_Data/QCD_train_discrete_pT_eta_phi.h5", key="df")
    val_data   = pd.read_hdf("/home/ew640340/Ph.D./Top_Quark_Tagging_Ref_Data/QCD_val_discrete_pT_eta_phi.h5", key="df")

    pT_train, eta_train, phi_train, counts_train = df_to_bin_tensors(train_data, max_particles=max_particles)
    pT_val, eta_val, phi_val, counts_val = df_to_bin_tensors(val_data, max_particles=max_particles)

    # Model
    embedder = ParticleEmbedder(emb_dim=256, pT_max_value=40, eta_max_value=30, phi_max_value=30, max_particles=max_particles, use_position_embedding=False, dropout=0.0)
    transformer = ParticleTransformer(num_pT_embeddings=42, num_eta_embeddings=32, num_phi_embeddings=32, emb_dim=256, n_layers=8, n_heads=4, dropout=0.1)

    save_dir = "/home/ew640340/Ph.D./Foundation_Models/checkpoints"

    history_df, best_path = train_and_evaluate(
        embedder=embedder, transformer=transformer,
        pT_train=pT_train, eta_train=eta_train, phi_train=phi_train, counts_train=counts_train,
        pT_val=pT_val, eta_val=eta_val, phi_val=phi_val, counts_val=counts_val,
        epochs=50,                           # Check: 1   |  Training: 50
        batch_size=128,                      # Check: 32  |  Training: 64+
        max_batches_to_check_train=None,      # Check: 50  |  Training: None 
        lr=5e-5,
        weight_decay=1e-4,
        train_max_real=train_max_real,
        device=device,
        max_grad_norm=1.0,
        enable_detect_anomaly=False,
        use_amp=True,
        save_dir=save_dir,
        save_best=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
        )

    print("\nBest model saved to:", best_path)
    history_df.to_hdf("/home/ew640340/Ph.D./Foundation_Models/train_val_history.h5", key="df", mode="w")


