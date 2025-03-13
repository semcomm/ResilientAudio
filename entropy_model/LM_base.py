"""
base class for LM model for vq indices of audio
"""
import torch
import torch.nn as nn
import numpy as np
import math

class LM_Base(nn.Module):
    def __init__(self, n_q, card, dim):
        """Language Model to estimate probabilities of each codebook entry.
        We predict all codebooks in parallel for a given time step.
        Args:
            n_q (int): number of codebooks.
            card (int): codebook cardinality.
            dim (int): transformer dimension.
            **kwargs: passed to `encodec.modules.transformer.StreamingTransformerEncoder`.
        """
        super().__init__()
        self.n_q = n_q
        self.card = card
        self.dim = dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)

    def get_coding_order(self, indices_size, context_tensor: torch.Tensor):
        """
        used in mode "fixed context"
        :param indices_size: [b,k,t]
        :param device:
        :return: coding_order: [K,T]
        """
        b,k,t = indices_size
        if context_tensor.ndim == 1:
            # temporal context only
            t_context = context_tensor.size(0)
            coding_order = torch.tile(context_tensor, dims=(k, math.ceil(t / t_context)))[:, :t]
        else:
            assert context_tensor.ndim == 2 and context_tensor.size(0) == k
            # stage and temporal context
            t_context = context_tensor.size(1)
            coding_order = torch.tile(context_tensor, dims=(t // t_context, ))[:, :t]

        return coding_order
