import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ParticleTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int = 256,
                 n_layers: int = 8,
                 n_heads: int = 4,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 pad_token: int = 0):
        """
        vocab_size: Größe des diskreten Particle-Dictionary (Anzahl Klassen)
        emb_dim: Embedding-Dimension (hier 256)
        n_layers: Anzahl TransformerEncoder-Layers (hier 8)
        n_heads: Anzahl Attention-Heads (hier 4)
        dim_feedforward: hidden dim im FFN (default 4*emb_dim)
        pad_token: Token-ID für Padding (ignore index beim Loss)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.pad_token = pad_token
        if dim_feedforward is None:
            dim_feedforward = emb_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=False)  # we'll pass (S, B, D)
        # TransformerEncoder mit Norm nach dem letzten Layer (wie in der Beschreibung)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                             norm=nn.LayerNorm(emb_dim))
        # zusätzlich LayerNorm & Dropout nach dem Encoder (wie gefordert)
        self.final_norm = nn.LayerNorm(emb_dim)
        self.final_dropout = nn.Dropout(dropout)

        # finaler FC-Head: Embedding -> Vocab logits
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self,
                seq_emb: torch.FloatTensor,
                src_key_padding_mask: torch.BoolTensor,
                causal_mask: torch.BoolTensor):
        """
        seq_emb: (B, S, D)
        src_key_padding_mask: (B, S) bool, True = padding
        causal_mask: (S, S) bool, True = mask (z.B. diagonal & future positions)
        returns:
           logits: (B, S, V)
        """
        B, S, D = seq_emb.shape
        assert D == self.emb_dim

        # Transformer erwartet (S, B, D) wenn batch_first=False
        x = seq_emb.transpose(0, 1).contiguous()  # -> (S, B, D)

        # Build attn_mask as float additive mask: positions to mask -> -inf
        # PyTorch erlaubt auch bool mask, aber additive float is robust:
        # attn_mask[i,j] will be added to attention scores (so large negative masks)
        attn_mask = torch.zeros((S, S), device=seq_emb.device, dtype=torch.float32)
        # causal_mask: True = mask (we map to -1e9)
        attn_mask = torch.where(causal_mask.to(seq_emb.device),
                                torch.tensor(float('-1e9'), device=seq_emb.device),
                                torch.tensor(0.0, device=seq_emb.device))

        # src_key_padding_mask: (B, S) bool True=pad -> matches Transformer API
        # run encoder
        enc = self.encoder(x,
                           mask=attn_mask,
                           src_key_padding_mask=src_key_padding_mask)  # (S, B, D)

        enc = enc.transpose(0, 1).contiguous()  # (B, S, D)
        enc = self.final_norm(enc)
        enc = self.final_dropout(enc)

        logits = self.head(enc)  # (B, S, V)
        return logits

# -------------------------
# Trainings-/Loss-Snippet (Beispiel)
# -------------------------
def train_example(model: nn.Module,
                  train_loader: DataLoader,
                  epochs: int = 50,
                  device: torch.device = torch.device('cpu'),
                  lr_init: float = 5e-4,
                  lr_final: float = 1e-6,
                  pad_token: int = 0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    # Cosine schedule mit eta_min = lr_final
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_final)
    # CrossEntropyLoss erwartet (N, C) logits und (N,) targets; ignore_index für gepadete Positionen
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            # Beispiel: angenommen Dataset liefert (seq_emb, src_key_padding_mask, causal_mask, target_tokens)
            seq_emb, src_key_padding_mask, causal_mask, target = batch
            seq_emb = seq_emb.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            causal_mask = causal_mask.to(device)
            target = target.to(device)  # (B, S) LongTensor with token ids, pad positions = pad_token

            optimizer.zero_grad()
            logits = model(seq_emb, src_key_padding_mask, causal_mask)  # (B, S, V)

            B, S, V = logits.shape
            logits_flat = logits.view(B * S, V)
            target_flat = target.view(B * S)

            loss = criterion(logits_flat, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * seq_emb.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model

# -------------------------
# Minimales Usage-Beispiel (Integration mit deinem Embedder)
# -------------------------
if __name__ == "__main__":
    # --- Annahmen / Platzhalter ---
    # Du musst "vocab_size" definieren: Anzahl der diskreten Partikel in deinem Dictionary.
    # Außerdem brauchst du target-Token pro Sequenzposition (z.B. Kombination aus pT/eta/phi -> single id).
    vocab_size = 5000   # <-- hier anpassen
    pad_token = 0

    # beispielhafte Dummy-Daten, ersetzbar durch echte seq_emb & masks aus deinem Embedder:
    # seq_emb: (B, S, D), src_key_padding_mask: (B, S), causal_mask: (S, S)
    # target_tokens: (B, S) long tensor with pad_token where to ignore
    B = 16
    # benutze deinen embedder, z.B.:
    # seq_emb = embedder(pT_t, eta_t, phi_t, counts)   # (B, S, D)
    # src_key_padding_mask, causal_mask = embedder.make_masks(pT_t, counts)

    # Für dieses Demo-File mache ich Dummy-Tensors:
    S = 1 + 50 + 1  # START + N + STOP wie in deinem Embedder
    D = 256
    seq_emb = torch.randn(B, S, D)
    src_key_padding_mask = torch.zeros(B, S, dtype=torch.bool)
    causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool), diagonal=0)  # j>=i masked

    # Ziel-Tokens dummy
    target_tokens = torch.randint(0, vocab_size, (B, S), dtype=torch.long)
    # setze padding positions in target falls src_key_padding_mask True:
    target_tokens[src_key_padding_mask] = pad_token

    ds = TensorDataset(seq_emb, src_key_padding_mask, causal_mask, target_tokens)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    model = ParticleTransformer(vocab_size=vocab_size, emb_dim=D, n_layers=8, n_heads=4, dropout=0.1, pad_token=pad_token)
    train_example(model, loader, epochs=2, device=torch.device('cpu'), lr_init=5e-4, lr_final=1e-6, pad_token=pad_token)




