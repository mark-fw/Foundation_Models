
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple

# --------------------------
# DF -> Tensors (liefert auch counts der realen Teilchen)
# --------------------------
def df_to_bin_tensors(df: pd.DataFrame, max_particles: int = 50, device: torch.device = torch.device('cpu')) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Liefert:
      pT_t, eta_t, phi_t: LongTensors shape (B, N) mit Werten (-1, 0..max)
      counts_t: LongTensor shape (B,) mit Anzahl realer Teilchen pro Event (nicht größer als N)
    Verhalten:
      - Trunkiert auf die ersten `max_particles` Teilchen (N).
      - Fehlende Werte bleiben -1.
      - counts zählt reale Teilchen über alle vorhandenen Spalten, wird aber auf N geklippt.
    """
    pattern = re.compile(r'^(pT|eta|phi)_bin_(\d+)$')
    idxs = set()
    for col in df.columns:
        m = pattern.match(col)
        if m:
            idxs.add(int(m.group(2)))
    existing_max_idx = max(idxs) if idxs else -1
    original_num_slots = existing_max_idx + 1

    batch_size = len(df)
    N = max_particles

    # prepare arrays with -1 padding
    pT_arr = np.full((batch_size, N), -1, dtype=np.int64)
    eta_arr = np.full((batch_size, N), -1, dtype=np.int64)
    phi_arr = np.full((batch_size, N), -1, dtype=np.int64)

    for i in range(N):
        pt_col = f"pT_bin_{i}"
        eta_col = f"eta_bin_{i}"
        phi_col = f"phi_bin_{i}"
        if pt_col in df.columns:
            pT_arr[:, i] = df[pt_col].fillna(-1).astype(np.int64).values
        if eta_col in df.columns:
            eta_arr[:, i] = df[eta_col].fillna(-1).astype(np.int64).values
        if phi_col in df.columns:
            phi_arr[:, i] = df[phi_col].fillna(-1).astype(np.int64).values

    # counts: benutze alle vorhandenen pT_bin_j Spalten (falls vorhanden), clip auf N
    if original_num_slots > 0:
        all_pT = np.full((batch_size, original_num_slots), -1, dtype=np.int64)
        for j in range(original_num_slots):
            col = f"pT_bin_{j}"
            if col in df.columns:
                all_pT[:, j] = df[col].fillna(-1).astype(np.int64).values
        counts = np.sum(all_pT != -1, axis=1).astype(np.int64)
    else:
        counts = np.zeros(batch_size, dtype=np.int64)

    # clip counts to N (max real particles we consider)
    counts = np.minimum(counts, N)

    # convert to tensors
    pT_t = torch.from_numpy(pT_arr).long().to(device)
    eta_t = torch.from_numpy(eta_arr).long().to(device)
    phi_t = torch.from_numpy(phi_arr).long().to(device)
    counts_t = torch.from_numpy(counts).long().to(device)

    return pT_t, eta_t, phi_t, counts_t


# --------------------------
# Embedder: START + up to N particles + STOP-slot (immer vorhanden) -> S = 1 + N + 1
# --------------------------
class ParticleEmbedder(nn.Module):
    def __init__(self,
                 emb_dim: int = 256,
                 pT_max_value: int = 40,
                 eta_max_value: int = 30,
                 phi_max_value: int = 30,
                 max_particles: int = 50,
                 use_position_embedding: bool = False,
                 dropout: float = 0.0):
        """
        max_particles = N (nur reale Teilchen zählen)
        Sequenzlänge S = 1 (START) + N (part slots) + 1 (STOP slot) = N + 2
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_particles = max_particles
        self.S = 1 + max_particles + 1  # START + N + STOP

        pT_slots = pT_max_value + 2
        eta_slots = eta_max_value + 2
        phi_slots = phi_max_value + 2

        self.pT_emb = nn.Embedding(num_embeddings=pT_slots, embedding_dim=emb_dim, padding_idx=0)
        self.eta_emb = nn.Embedding(num_embeddings=eta_slots, embedding_dim=emb_dim, padding_idx=0)
        self.phi_emb = nn.Embedding(num_embeddings=phi_slots, embedding_dim=emb_dim, padding_idx=0)

        self.use_position_embedding = use_position_embedding
        if use_position_embedding:
            # Positions for real particle slots are 1..N; we do not assign position to START or STOP here
            self.pos_emb = nn.Embedding(max_particles + 1, emb_dim)

        # learned START token (index 0 in sequence)
        self.start_token = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        # learned STOP token (we will place it at index 1+R for R < N; otherwise it's padding)
        self.stop_token = nn.Parameter(torch.randn(1, emb_dim) * 0.02)

        self.layernorm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, pT_bins: torch.LongTensor, eta_bins: torch.LongTensor, phi_bins: torch.LongTensor, counts: torch.LongTensor) -> torch.FloatTensor:
        """
        Inputs:
          pT_bins, eta_bins, phi_bins: (B, N) LongTensors in original range (-1 = padding)
          counts: (B,) LongTensor with number of real particles per event clipped to N
        Output:
          seq: (B, S, D) with S = 1 + N + 1; START at 0, real particles placed left-aligned,
               STOP inserted after last real particle at index (1 + counts[b]) only when counts[b] < N,
               remaining positions are padding (zero embeddings) and should be masked later.
        """
        B, N = pT_bins.shape
        assert N == self.max_particles, "pT_bins must have shape (B, max_particles)"
        device = pT_bins.device
        S = self.S
        D = self.emb_dim

        # shift indices for embeddings: -1 -> 0 (padding_idx)
        pT_idx = torch.clamp(pT_bins + 1, min=0, max=self.pT_emb.num_embeddings - 1)
        eta_idx = torch.clamp(eta_bins + 1, min=0, max=self.eta_emb.num_embeddings - 1)
        phi_idx = torch.clamp(phi_bins + 1, min=0, max=self.phi_emb.num_embeddings - 1)

        e_pT = self.pT_emb(pT_idx)    # (B, N, D)
        e_eta = self.eta_emb(eta_idx)
        e_phi = self.phi_emb(phi_idx)

        e_particles = e_pT + e_eta + e_phi  # (B, N, D)

        if self.use_position_embedding:
            pos_idx = torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, N)  # 1..N
            e_particles = e_particles + self.pos_emb(pos_idx)

        # ---- Aufbau der finalen Sequenz via Scatter (rechts-shift für padding nach STOP) ----
        # seq init zeros
        seq = torch.zeros((B, S, D), dtype=e_particles.dtype, device=device)

        # START token an index 0
        seq[:, 0, :] = self.start_token.view(1, D)

        # Berechne für jedes Batch die Zielfelder (dest positions) für die N particle slots,
        # so dass alle realen Partikel links bleiben und padding nach STOP geschoben wird:
        # dest[b, j] = 1 + j + (j >= counts[b] ? 1 : 0)
        arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # (B, N)
        counts_exp = counts.unsqueeze(1).expand(B, N)                     # (B, N)
        shift = (arange >= counts_exp).long()                             # (B, N) 0 or 1
        dest = 1 + arange + shift  # values in [1 .. N+1] (since shift can add 1)
        # dest has shape (B, N), values within 1..(N+1) inclusive

        # scatter particle embeddings into seq at the computed dest positions
        # expand dest to have D last dimension for scatter
        dest_idx = dest.unsqueeze(-1).expand(B, N, D)  # (B, N, D)
        seq = seq.scatter(dim=1, index=dest_idx, src=e_particles)

        # Stop token: only for those events where counts[b] < N
        stop_needed = (counts < N)  # (B,)
        if stop_needed.any():
            stop_pos = 1 + counts  # (B,) position where STOP should go (index in seq)
            # apply stop_token to those positions
            # prepare expanded stop_token and an index tensor
            stop_positions = stop_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)  # (B,1,D)
            # we can't directly scatter a (B,1,D) src into seq with different indices per batch easily,
            # so do per-batch assignment in vectorized manner:
            batch_idx = torch.arange(B, device=device)
            mask_stop = stop_needed  # bool mask
            seq[batch_idx[mask_stop], stop_pos[mask_stop], :] = self.stop_token.view(1, D)

        # Apply LayerNorm & dropout
        seq = self.layernorm(seq)
        seq = self.dropout(seq)
        return seq  # (B, S, D)

    def make_masks(self, pT_bins: torch.LongTensor, counts: torch.LongTensor) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Erzeugt:
         - src_key_padding_mask: (B, S) True = mask (padding)
             -> START (index 0): False
             -> real particle positions: False for j < counts[b], else (the positions after STOP) True
             -> STOP: False if counts[b] < N, else True (i.e., if no STOP because counts==N)
         - causal_mask: (S, S) True where attention should be masked (we mask j >= i),
            so that each position i attends only to j < i (START attends to none).
        """
        B, N = pT_bins.shape
        device = pT_bins.device
        S = self.S

        # original padding mask per particle slot (B, N): True where pT == -1 (original padding)
        orig_pad = (pT_bins == -1)  # True where padding in original slots

        # Build destination positions as in forward to map orig_pad into final seq positions
        arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        counts_exp = counts.unsqueeze(1).expand(B, N)
        shift = (arange >= counts_exp).long()
        dest = 1 + arange + shift  # (B, N) values in 1..N+1

        # initialize src_key_padding_mask True everywhere; we'll clear non-padding positions
        src_key_padding_mask = torch.ones((B, S), dtype=torch.bool, device=device)

        # START not masked
        src_key_padding_mask[:, 0] = False

        # scatter original padding flags into dest positions:
        # for positions where orig_pad == True, set corresponding dest positions to True (remain masked)
        # for positions where orig_pad == False (real particles), set dest positions to False (not masked)
        # We can start by setting all dest positions to True and then clear where orig_pad == False
        # Efficient vectorized assignment:
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)  # (B, N)
        dest_flat = dest  # (B, N)
        # set masked=True where orig_pad True
        src_key_padding_mask[batch_idx[orig_pad], dest_flat[orig_pad]] = True
        # set masked=False where orig_pad False (real particles)
        src_key_padding_mask[batch_idx[~orig_pad], dest_flat[~orig_pad]] = False

        # STOP handling: if counts[b] < N -> stop at pos = 1 + counts[b] and must NOT be masked
        stop_needed = (counts < N)
        if stop_needed.any():
            stop_pos = 1 + counts  # (B,)
            batch_idx_simple = torch.arange(B, device=device)
            src_key_padding_mask[batch_idx_simple[stop_needed], stop_pos[stop_needed]] = False

        # causal_mask: True where j >= i (including diagonal), so positions can only attend to earlier j < i
        causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device), diagonal=0)

        # Convert to float additive mask expected by nn.Transformer (S,S)
        attn_mask = torch.zeros(S, S, device=device, dtype=torch.float32)
        attn_mask[causal_mask] = float('-inf')  # or a large negative number

        return src_key_padding_mask, causal_mask, attn_mask


# --------------------------
# Beispiel: Verwendung
# --------------------------
if __name__ == "__main__":
    
    df = pd.read_hdf("/home/ew640340/Ph.D./Foundation_Models/val_discrete_pT_eta_phi.h5", key="df")
    training_data = df[df["is_signal_like"] == 1].reset_index(drop=True).copy().head(1000)

    device = torch.device('cpu')

    # --- Konfiguration ---
    N = 50    # max reale Teilchen 
    emb_dim = 256

    # DF -> Tensors + counts
    pT_t, eta_t, phi_t, counts = df_to_bin_tensors(training_data, max_particles=N, device=device)
    B = pT_t.shape[0]

    # Embedder
    embedder = ParticleEmbedder(emb_dim=emb_dim, pT_max_value=40, eta_max_value=30, phi_max_value=30,
                                max_particles=N, use_position_embedding=False, dropout=0.0)
    embedder.to(device)

    # Sequence embeddings: (B, S, D) with S = 1 + N + 1
    seq_emb = embedder(pT_t, eta_t, phi_t, counts)

    # Masks
    src_key_padding_mask, causal_mask, attn_mask = embedder.make_masks(pT_t, counts)
    # src_key_padding_mask: (B, S), causal_mask: (S, S)

    # Transformer Encoder (batch_first=True)
    encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=emb_dim,
                                               dropout=0.1, activation='gelu', batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8).to(device)

    out = transformer_encoder(seq_emb, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
    start_repr = out[:, 0, :]

    print("S (sequence length) =", embedder.S)
    print("seq_emb shape:", seq_emb.shape)
    print("src_key_padding_mask shape:", src_key_padding_mask.shape)
    print("causal_mask shape:", causal_mask.shape)
    print("transformer output shape:", out.shape)
    print("counts:", counts)
