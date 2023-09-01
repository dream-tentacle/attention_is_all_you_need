import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from PE import PE
from encoder import Encoder
from decoder import Decoder


class EncoderDecoder(nn.Module):
    # only contains encoder and decoder
    def __init__(self, layer_num, d_model, head, hidden_size, dropout=0.5) -> None:
        super().__init__()
        self.encoder = Encoder(layer_num, d_model, head, hidden_size, dropout)
        self.decoder = Decoder(layer_num, d_model, head, hidden_size, dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_memory, tgt_mask)
        return output


class Transformer(nn.Module):
    def __init__(
        self, layer_num, d_model, head, hidden_size, vocab, dropout=0.5
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.enc_embed = nn.Embedding(vocab, d_model)
        self.dec_embed = nn.Embedding(vocab, d_model)
        self.encdec = EncoderDecoder(layer_num, d_model, head, hidden_size, dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_size), nn.ReLU(), nn.Linear(hidden_size, vocab)
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        """the whole forward for transformer

        Args:
            src: the source sentences, shape of [B,src_len]
            tgt: the target sentences, shape of [B,tgt_len]
            src_mask: the mask for source sentences' padding, shape of [B,src_len,src_len],
                    and for one piece in a batch, all is the same in a column
            tgt_mask: the mask for future words, shape of [tgt_len,tgt_len]
        """
        src = self.enc_embed(src) + torch.stack(
            [PE(pos, self.d_model) for pos in range(src.shape[-1])]
        ).to(src.device)
        tgt = self.dec_embed(tgt) + torch.stack(
            [PE(pos, self.d_model) for pos in range(tgt.shape[-1])]
        ).to(tgt.device)
        output = self.encdec(src, tgt, src_mask, tgt_mask)
        output = self.fc(output)
        return output
