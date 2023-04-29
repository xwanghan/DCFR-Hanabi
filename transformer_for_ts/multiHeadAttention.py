from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * self._h)
        self._W_k = nn.Linear(d_model, q * self._h)
        self._W_v = nn.Linear(d_model, v * self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h * v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                word_length: torch.Tensor,
                decoder: int, ) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        word_length:
        decoder:


        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        # (batch_size, K, d_model) to (batch_size, head, K, d_model//head)
        queries = torch.stack(self._W_q(query).chunk(self._h, dim=-1), dim=1)
        keys = torch.stack(self._W_k(key).chunk(self._h, dim=-1), dim=1)
        values = torch.stack(self._W_v(value).chunk(self._h, dim=-1), dim=1)

        # Scaled Dot Product
        # (batch_size, head, Kq, d_model//head) * (batch_size, head,d_model//head, K)
        # (batch_size, Kq, K)
        self._scores = (queries @ keys.transpose(2, 3)) / np.sqrt(K)

        # print(self._scores)
        # Compute local map mask

        attention_mask = generate_local_map_mask(K, word_length, decoder, device=self._scores.device)
        self._scores = self._scores.masked_fill(attention_mask, -1e9)

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        # print(query.shape, keys.shape, self._scores.size(), values.size(), self._scores)

        # (batch_size, head, Kq, K) * (batch_size, K, d_model//head)
        # (batch_size, head, Kq, d_model//head)
        attention = self._scores @ values

        # Concatenat the heads
        # (batch_size, head, Kq, d_model//head) to (batch_size, Kq, d_model)
        attention_heads = torch.cat(attention.chunk(self._h, dim=1), dim=-1).squeeze(dim=1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


def generate_local_map_mask(chunk_size: int,
                            word_length: torch.Tensor,
                            decoder: int,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    word_length:
    decoder:

    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    batch_size = word_length.shape[0]
    if decoder:
        local_map = torch.ones((batch_size, 1, chunk_size, chunk_size))
        for i in range(batch_size):
            local_map[i][0][:word_length[i], 1] = torch.zeros((word_length[i]))
    else:
        local_map = torch.ones((batch_size, 1, chunk_size, chunk_size))
        for i in range(batch_size):
            local_map[i][0][:word_length[i], :word_length[i]] = torch.zeros((word_length[i], word_length[i]))

    return local_map.type(torch.BoolTensor).to(device)


