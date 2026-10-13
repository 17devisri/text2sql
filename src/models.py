import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module from scratch.
    Computes Scaled Dot-Product Attention across h heads in parallel.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # 1) Linear projections and reshape to (batch, heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask shape expected: (batch, 1, seq_len_q, seq_len_k) or broadcastable
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3) Multiply by values and concatenate heads
        out = torch.matmul(attn_weights, v)  # (batch, heads, seq_len_q, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4) Final linear projection
        return self.w_o(out), attn_weights

class PositionWiseFeedForward(nn.Module):
    """Feed-forward sub-layer with GELU activation and dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Single layer of Transformer Encoder (Self-Attention + FFN)."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture for improved training stability   
        norm_src = self.norm1(src)
        attn_out, _ = self.self_attn(norm_src, norm_src, norm_src, src_mask)
        src = src + self.dropout(attn_out)

        norm_src2 = self.norm2(src)
        ffn_out = self.ffn(norm_src2)
        src = src + self.dropout(ffn_out)
        return src

class TransformerDecoderLayer(nn.Module):
    """Single layer of Transformer Decoder (Causal Self-Attention + Cross-Attention + FFN)."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Masked Self-Attention
        norm_tgt = self.norm1(tgt)
        attn_out, _ = self.self_attn(norm_tgt, norm_tgt, norm_tgt, tgt_mask)
        tgt = tgt + self.dropout(attn_out)

        # 2. Cross-Attention with Encoder outputs (memory)
        norm_tgt2 = self.norm2(tgt)
        cross_out, _ = self.cross_attn(norm_tgt2, memory, memory, memory_mask)
        tgt = tgt + self.dropout(cross_out)

        # 3. Position-wise Feed Forward
        norm_tgt3 = self.norm3(tgt)
        ffn_out = self.ffn(norm_tgt3)
        tgt = tgt + self.dropout(ffn_out)
        return tgt

class TransformerSeq2Seq(nn.Module):
    """
    End-to-End Seq2Seq Transformer built from scratch in PyTorch for Text-to-SQL.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_final_norm = nn.LayerNorm(d_model)

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_final_norm = nn.LayerNorm(d_model)

        # Final projection to target vocabulary
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create mask for source padding tokens: (batch, 1, 1, src_len)."""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create combined mask for target: padding mask AND causal look-ahead mask."""
        batch_size, tgt_len = tgt.size()
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)

        # Lower triangular causal mask (1 where allowed to attend, 0 where disallowed)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)

        return pad_mask & subsequent_mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass source through embedding and encoder layers."""
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_final_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pass target and encoder representations through decoder layers."""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        for layer in self.decoder_layers:
            x = layer(tgt=x, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        x = self.decoder_final_norm(x)
        return self.fc_out(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Full forward pass for training with teacher forcing."""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return output

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) Attention mechanism for RNN Seq2Seq."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_enc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # Bidirectional encoder
        self.w_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # decoder_hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        dec_proj = self.w_dec(decoder_hidden).unsqueeze(1)  # (batch, 1, hidden_dim)
        enc_proj = self.w_enc(encoder_outputs)              # (batch, seq_len, hidden_dim)

        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(2)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1)  # (batch, seq_len)

class LSTMSeq2Seq(nn.Module):
    """Bidirectional LSTM Encoder-Decoder with Bahdanau Attention."""
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim

        # Encoder
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.enc_to_dec_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.enc_to_dec_c = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim)
        self.decoder_cell = nn.LSTMCell(embed_dim + hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2 + embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # Encode
        src_mask = (src != self.pad_idx)
        src_embedded = self.dropout(self.src_embed(src))
        enc_outputs, (h_n, c_n) = self.encoder(src_embedded)

        # Combine bidirectional states for initial decoder state
        last_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        last_c = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        dec_h = torch.tanh(self.enc_to_dec_h(last_h))
        dec_c = torch.tanh(self.enc_to_dec_c(last_c))

        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=src.device)
        input_token = tgt[:, 0]  # <sos> token

        for t in range(1, tgt_len):
            emb = self.dropout(self.tgt_embed(input_token))
            attn_weights = self.attention(dec_h, enc_outputs, mask=src_mask)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

            lstm_in = torch.cat([emb, context], dim=-1)
            dec_h, dec_c = self.decoder_cell(lstm_in, (dec_h, dec_c))

            out = self.fc_out(torch.cat([dec_h, context, emb], dim=-1))
            outputs[:, t - 1] = out

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = out.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs
