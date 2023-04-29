import torch
import torch.nn as nn

from transformer_for_ts.encoder import Encoder
from transformer_for_ts.decoder import Decoder


class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer_for_ts model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is
        ``None``.
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 d_ff: int,
                 dropout: float = 0.3,
                 compute_value: int = 0):
        """Create transformer_for_ts structure from Encoder and Decoder blocks."""
        super().__init__()

        self.compute_value = compute_value

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      d_ff,
                                                      dropout=dropout) for _ in range(N)])
        '''self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      d_ff,
                                                      dropout=dropout) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)'''
        self._linear = nn.Linear(d_model, d_output)

        self.name = 'transformer_for_ts'

    def forward(self, x: torch.Tensor,  word_length: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer_for_ts

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        word_length:
            :


        Returns
        -------
            Output tensor with shape (batch_size, d_output).
        """

        # no Embeddin module
        encoding = x

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, word_length)

        # Decoding stack
        '''decoding = encoding

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding, word_length)'''

        # Output module
        output = self._linear(encoding)
        output = torch.sum(output, dim=1)
        if self.compute_value:
            return output
        else:
            output = torch.softmax(output, dim=1)
            return output
