import torch
from torch import nn
import torch.nn.functional as F

class RelativeAttention(nn.Module):
    def __init__(self, d_model, num_heads, window=8, dropout=0.2, reverse=False, d_output=None, fudge_output=False):
        super().__init__()
        if d_output is None:
            d_output = d_model

        d_head, remainder = divmod(d_output, num_heads)
        if remainder:
            if fudge_output:
                d_head = d_head + 1
                logger.debug("Relative attn: %d %% %d != 0, updating d_output to %d", d_output, num_heads, num_heads * d_head)
                d_output = num_heads * d_head
            else:
                raise ValueError("incompatible `d_model` and `num_heads`")
        self.window = window
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.d_output = d_output
        self.key = nn.Linear(d_model, d_output)
        # the bias for query all gets trained to 0 anyway
        self.query = nn.Linear(d_model, d_output, bias=False)
        self.value = nn.Linear(d_model, d_output, bias=False)
        # initializing value with eye seems to hurt!
        #nn.init.eye_(self.value.weight)

        self.dropout = nn.Dropout(dropout)
        self.position = nn.Parameter(torch.randn(1, 1, d_head, window, 1))

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(window, window), diagonal=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            "flipped_mask",
            torch.flip(self.mask, (-1,))
        )

        self.reverse = reverse

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("Incompatible input")

        if self.reverse:
            x = torch.flip(x, (1,))

        orig_seq_len = seq_len
        if seq_len < self.window:
            zeros = torch.zeros((x.shape[0], self.window - seq_len, x.shape[2]), dtype=x.dtype, device=x.device)
            x = torch.cat((x, zeros), axis=1)
            seq_len = self.window

        # k.shape = (batch_size, num_heads, d_head, seq_len)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)

        # v.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)

        # q.shape = (batch_size, num_heads, d_head, seq_len)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # q.shape = (batch_size, num_heads, d_head, window, seq_len)
        q = self.skew_repeat(q)
        q = q + self.position

        # qk.shape = (batch_size, num_heads, d_head, window, seq_len)
        qk = torch.einsum('bndws,bnds->bndws', q, k)

        # mask out the padding spaces at the end
        # can only attend to spots that aren't padded
        if orig_seq_len < seq_len:
            # mask out the part of the sentence which is empty
            shorter_mask = self.flipped_mask[:, :, :, :, -orig_seq_len:]
            qk[:, :, :, :, :orig_seq_len] = qk[:, :, :, :, :orig_seq_len].masked_fill(shorter_mask == 1, float("-inf"))
            qk = qk[:, :, :, :orig_seq_len, :orig_seq_len]
        else:
            qk[:, :, :, :, -self.window:] = qk[:, :, :, :, -self.window:].masked_fill(self.flipped_mask == 1, float("-inf"))
        qk = F.softmax(qk, dim=3)

        # v.shape = (batch_size, num_heads, d_head, window, seq_len)
        v = self.skew_repeat(v)
        if orig_seq_len < seq_len:
            v = v[:, :, :, :orig_seq_len, :orig_seq_len]
        # result.shape = (batch_size, num_heads, d_head, orig_seq_len)
        result = torch.einsum('bndws,bndws->bnds', qk, v)
        # batch_size, orig_seq_len, d_output
        result = result.reshape(batch_size, self.d_output, orig_seq_len).transpose(1, 2)

        if self.reverse:
            result = torch.flip(result, (1,))

        return self.dropout(result)

    def skew_repeat(self, q):
        # make stripes that look like this
        # (seq_len 5, window 3)
        #   1 2 3 4 5
        #   1 2 3 4 5
        #   1 2 3 4 5
        q = q.unsqueeze(4).repeat(1, 1, 1, 1, self.window).transpose(3, 4)
        # now the stripes look like
        #   1 2 3 4 5
        #   0 2 3 4 5
        #   0 0 3 4 5
        q[:, :, :, :, :self.window] = q[:, :, :, :, :self.window].masked_fill(self.mask == 1, 0)
        q_shape = list(q.shape)
        q_new_shape = list(q.shape)[:-2] + [-1]
        q = q.reshape(q_new_shape)
        zeros = torch.zeros_like(q[:, :, :, :1])
        zeros = zeros.repeat(1, 1, 1, self.window)
        q = torch.cat((q, zeros), axis=-1)
        q_new_shape = q_new_shape[:-1] + [self.window, -1]
        # now the stripes look like
        #   1 2 3 4 5
        #   2 3 4 5 0
        #   3 4 5 0 0
        # q.shape = (batch_size, num_heads, d_head, window, seq_len)
        q = q.reshape(q_new_shape)[:, :, :, :, :-1]
        return q
