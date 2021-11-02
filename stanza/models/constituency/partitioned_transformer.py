"""
Transformer with partitioned content and position features.

See section 3 of https://arxiv.org/pdf/1805.01052.pdf
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = torch.empty(
                (input.size(0), input.size(-1)),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[:, None, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None
        else:
            return grad_output, None, None, None


class FeatureDropout(nn.Dropout):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
            x_c = FeatureDropoutFunction.apply(x_c, self.p, self.training, self.inplace)
            x_p = FeatureDropoutFunction.apply(x_p, self.p, self.training, self.inplace)
            return x_c, x_p
        else:
            return FeatureDropoutFunction.apply(x, self.p, self.training, self.inplace)


class PartitionedReLU(nn.ReLU):
    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        return super().forward(x_c), super().forward(x_p)


class PartitionedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_c = nn.Linear(in_features // 2, out_features // 2, bias)
        self.linear_p = nn.Linear(in_features // 2, out_features // 2, bias)

    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)

        out_c = self.linear_c(x_c)
        out_p = self.linear_p(x_p)
        return out_c, out_p


class PartitionedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model, n_head, d_qkv, attention_dropout=0.1, initializer_range=0.02
    ):
        super().__init__()

        self.w_qkv_c = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, d_qkv // 2))
        self.w_qkv_p = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, d_qkv // 2))
        self.w_o_c = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model // 2))
        self.w_o_p = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model // 2))

        bound = math.sqrt(3.0) * initializer_range
        for param in [self.w_qkv_c, self.w_qkv_p, self.w_o_c, self.w_o_p]:
            nn.init.uniform_(param, -bound, bound)
        self.scaling_factor = 1 / d_qkv ** 0.5

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        qkv_c = torch.einsum("btf,hfca->bhtca", x_c, self.w_qkv_c)
        qkv_p = torch.einsum("btf,hfca->bhtca", x_p, self.w_qkv_p)
        q_c, k_c, v_c = [c.squeeze(dim=3) for c in torch.chunk(qkv_c, 3, dim=3)]
        q_p, k_p, v_p = [c.squeeze(dim=3) for c in torch.chunk(qkv_p, 3, dim=3)]
        q = torch.cat([q_c, q_p], dim=-1) * self.scaling_factor
        k = torch.cat([k_c, k_p], dim=-1)
        v = torch.cat([v_c, v_p], dim=-1)
        dots = torch.einsum("bhqa,bhka->bhqk", q, k)
        if mask is not None:
            dots.data.masked_fill_(~mask[:, None, None, :], -float("inf"))
        probs = F.softmax(dots, dim=-1)
        probs = self.dropout(probs)
        o = torch.einsum("bhqk,bhka->bhqa", probs, v)
        o_c, o_p = torch.chunk(o, 2, dim=-1)
        out_c = torch.einsum("bhta,haf->btf", o_c, self.w_o_c)
        out_p = torch.einsum("bhta,haf->btf", o_p, self.w_o_p)
        return out_c, out_p


class PartitionedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_qkv,
        d_ff,
        ff_dropout=0.1,
        residual_dropout=0.1,
        attention_dropout=0.1,
        activation=PartitionedReLU(),
    ):
        super().__init__()
        self.self_attn = PartitionedMultiHeadAttention(
            d_model, n_head, d_qkv, attention_dropout=attention_dropout
        )
        self.linear1 = PartitionedLinear(d_model, d_ff)
        self.ff_dropout = FeatureDropout(ff_dropout)
        self.linear2 = PartitionedLinear(d_ff, d_model)

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.residual_dropout_attn = FeatureDropout(residual_dropout)
        self.residual_dropout_ff = FeatureDropout(residual_dropout)

        self.activation = activation

    def forward(self, x, mask=None):
        residual = self.self_attn(x, mask=mask)
        residual = torch.cat(residual, dim=-1)
        residual = self.residual_dropout_attn(residual)
        x = self.norm_attn(x + residual)
        residual = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        residual = torch.cat(residual, dim=-1)
        residual = self.residual_dropout_ff(residual)
        x = self.norm_ff(x + residual)
        return x


class PartitionedTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(n_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class ConcatPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model // 2))
        nn.init.normal_(self.timing_table)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        timing = self.timing_table[None, : x.shape[1], :]
        x, timing = torch.broadcast_tensors(x, timing)
        out = torch.cat([x, timing], dim=-1)
        out = self.norm(out)
        return out
