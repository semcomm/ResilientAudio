import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F
from entropy_model.LM_base import LM_Base
import modules as m
import math
import random
import numpy as np
from modules.conformer import Conformer
from einops import rearrange
from einops.layers.torch import Rearrange, EinMix
from typing import List, Tuple
from warnings import warn
import time
import constriction


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        # print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)


def cosine_schedule(t):
    """
    :param t:  uniformly sampling from [0,1]
    :return:
    """
    return torch.cos(t * math.pi * 0.5)


class LMConformer(LM_Base):
    def __init__(self, n_q, card, dim,
                 n_q_options: List[int] = None,
                 base_layer_n_q: int = 3,
                 heads=4, linear_units=4096, num_blocks=12, positionwise_conv_kernel_size=5, **kwargs):

        super(LMConformer, self).__init__(n_q, card, dim)
        self.ignore_index = -1
        num_codes_with_mask = card + 1
        self.mask_token_id = num_codes_with_mask
        self.mask_upper_level = num_codes_with_mask + 1
        self.num_special_tokens = 2

        # audio code： 1，2，，，，card
        # mask_token: card + 1
        # mask_upper_level: card + 2

        self.base_layer_n_q = base_layer_n_q

        self.lm = Conformer(
            attention_dim=dim,
            attention_heads=heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size
        )
        print(f'LM config: dim {dim} heads {heads} blocks {num_blocks}')
        self.emb = nn.ModuleList(
            [nn.Embedding(card + self.num_special_tokens, dim) for _ in range(n_q)])

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * n_q),
            Rearrange('b n (h d) -> b (n h) d', h=n_q),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
            Rearrange('b (n q) d -> b n q d', q=n_q)
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            EinMix(
                'b n q d -> b n q l',
                weight_shape='q d l',
                bias_shape='q l',
                q=n_q,
                l=card,
                d=dim
            )
        )

        self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)

        stage_weight_factor = np.linspace(5.0, 1.0, self.n_q)
        self.register_buffer('stage_weight_factor', torch.as_tensor(stage_weight_factor))

        self.apply(weights_init)

        self.gamma = self.gamma_func("sine")  # linear

        if n_q_options is not None:
            # for group optimization
            n_q_start = n_q_options[:-1]
            n_q_end = n_q_options[1:]
            self.n_q_options = np.array(n_q_options)  # [3, 6, 12, 24, 36]  -->target_bandwidth=[1.5, 3, 6, 12, 18]
            self.q_range_options = [[n_q_s, n_q_e] for n_q_s, n_q_e in
                                    zip(n_q_start, n_q_end)]  # finelayer q_range  [3,6],[6,12],[12,24],[24,36]
            print("LM q_range_options: ", self.q_range_options)

        self.count_params()

    def count_params(self):
        print(f'LM params = {sum([p.numel() for p in self.parameters()]) / 1e6 :.3f} M')
        print(f'LM emb params = {sum([p.numel() for p in self.emb.parameters()]) / 1e6 :.3f} M')
        print(f'LM lm params = {sum([p.numel() for p in self.lm.parameters()]) / 1e6 :.3f} M')
        print(f'LM heads params = {sum([p.numel() for p in self.heads.parameters()]) / 1e6 :.3f} M')
        print(f'LM to_logits params = {sum([p.numel() for p in self.to_logits.parameters()]) / 1e6 :.3f} M')

    def tokens_to_logits(self, indices_btk, stage: int):
        """

        :param indices_btk:
        :param stage:
        :return: logits shape [b,t,k,c]
        """
        emb = None  # [b,t,d]
        for i in range(stage):
            if emb is None:
                emb = self.emb[i](indices_btk[:, :, i].unsqueeze(-1)).squeeze(-2)
            else:
                emb = emb + self.emb[i](indices_btk[:, :, i].unsqueeze(-1)).squeeze(-2)

        out, _ = self.lm(emb, None)  # [B, t, d]
        out = self.heads(out)  # [:, :, :self.n_q]
        logits = self.to_logits(out)  # [b, t, n_q, card]

        return logits, out

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        elif mode == 'sine':
            return lambda r: np.sin(r * np.pi / 2)
        elif mode == "cosfull":
            return lambda r: 0.5 * (1 + np.cos(r * np.pi))
        elif mode == "cubicmirror":
            return lambda r: 0.5 - 4 * (r - 0.5) ** 3
        else:
            raise NotImplementedError

    def fine_mask(self, code, unmask, q_start: int = None):
        """
        mask q_start and above
        :param code:  [b, k, t]
        :param unmask:  [b, k, t]  update to generate label where unmask is False
        :param t:
        :return:
        """
        q_start = self.base_layer_n_q if q_start is None else q_start
        code[:, q_start:, :] = self.mask_upper_level
        unmask[:, q_start:, :] = True  # to ignore loss
        return code, unmask

    def masking(self, codes_bkt, code_shape, base_layer: bool, q_range: Tuple[int, int] = None, mask_proportion=None,
                frames_mask: int = 0, lookahead_frames: int = 0):
        """
        random masking of codes_bkt base_layer masking or fine_layer masking with specific q_range
        base layer masking training for base layer concealment
        fine layer masking training for fine layer compression and fine layer prediction

        :param codes_bkt:
        :param code_shape:  (b,k,t)
        :param q_range: range of levels to pred (fine level), [0, 2):  level base and level base + 1 are the target
           if q is None, all fine tokens are predicted.  Valid only base_layer is False
        :return: masked_codes
                 label:  same shape as codes_bkt
        """
        # random mask 
        keep_prob = 1. - self.gamma(np.random.uniform()) if mask_proportion is None else 1. - mask_proportion
        r = math.floor(keep_prob * code_shape[-1])  
        sample = torch.rand(code_shape[0], code_shape[-1], device=codes_bkt.device).topk(r, dim=-1).indices
        # True for unmask(known codeword)
        unmask_bt = torch.zeros(code_shape[0], code_shape[-1],
                                dtype=torch.bool, device=codes_bkt.device).scatter_(dim=-1,
                                                                                    index=sample,
                                                                                    value=True)
        if frames_mask:
            if lookahead_frames > 0:
                unmask_bt[:, -(frames_mask + lookahead_frames):-lookahead_frames] = False
            else:
                unmask_bt[:, -frames_mask:] = False

        unmask_bkt = unmask_bt[:, None].repeat(1, code_shape[1], 1)
        codes_in = codes_bkt.clone()
        if base_layer:
            # mask base layer only and fine layer tokens all masked
            masked_codes = torch.where(unmask_bkt, codes_in, self.mask_token_id)
            masked_codes, unmask_bkt = self.fine_mask(masked_codes, unmask_bkt)  # will modify unmask_bkt in place
        else:
            # base layer of all frames is known(accurately) or base layer fec correct
            if q_range is not None:
                """ # conditioned on coarse tokens (q_range[0] layers including base and fine tokens) (for concealing)
                    # predict the upper q_range[1] - q_range[0] layers    
                """
                # groupwise masking for fine layers
                unmask_bkt[:, :q_range[0]] = True
                masked_codes = torch.where(unmask_bkt, codes_in, self.mask_token_id)
                masked_codes, mask_bkt = self.fine_mask(masked_codes, unmask_bkt,
                                                        q_start=q_range[1])
                # token to logits stage == q_range[1]
                # predict q_range[0] to q_range[1] conditioned on q_range[0] layers of codes
            else:

                # fine tokens (all stages from base_layer_n_q to K are the target)
                unmask_bkt[:, :self.base_layer_n_q] = True
                masked_codes = torch.where(unmask_bkt, codes_in, self.mask_token_id)

        label = torch.where(unmask_bkt, self.ignore_index, codes_bkt)  # count loss on False positions

        return masked_codes, label

    def forward(self, streaming, indices, base_layer: bool, D_index: int = None, mask_proportion: float = None,
                return_loss=True, weighted_loss=False, q_range=None, steps_cnt_loss: int = 5,
                lookahead_frames: int = 0):
        # if streaming:
        #     return self.forward_streaming(indices, base_layer, D_index, mask_proportion,
        #                                   return_loss=return_loss, weighted_loss=weighted_loss,
        #                                   steps_cnt_loss=steps_cnt_loss,
        #                                   lookahead_frames=lookahead_frames)
        # else:
        return self.forward_nonstreaming(indices, base_layer, D_index, mask_proportion,
                                         return_loss=return_loss, weighted_loss=weighted_loss)

    def forward_nonstreaming(self, indices,
                             base_layer: bool, D_index: int,
                             mask_proportion: float,
                             return_loss,
                             weighted_loss):
        B, K, T = indices.shape

        if base_layer:
            Q = self.base_layer_n_q
        else:
            if random.random() > 0.5:  # random choose from (2,8) for entropy modeling and (4,8) for concealing
                q_range = [self.base_layer_n_q, K]
            else:
                q_range = [self.q_range_options[D_index - 1][0], K]
            Q = K

        masked_codes, labels = self.masking(indices, (B, K, T), base_layer=base_layer,
                                            mask_proportion=mask_proportion,
                                            q_range=q_range if not base_layer else None)  # all fine layer counted in
        masked_codes = rearrange(masked_codes, 'b k t -> b t k')

        logits, out = self.tokens_to_logits(masked_codes, stage=Q)  # [b,t,n_q,card]   sum up Q stages embeddings
        # # indices.shape[2] == K <= self.n_q

        logits = logits[:, :, :Q]
        labels = labels[:, :Q]  # b,k,t
        logits = rearrange(logits, 'b n k c -> b c k n')

        if return_loss:
            if weighted_loss:
                loss_bkt = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.ignore_index,
                    reduction='none'
                )
                loss = loss_bkt * self.stage_weight_factor[None, :Q, None]
                loss = loss[labels != self.ignore_index].mean()
            else:
                loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.ignore_index
                )
            return loss, logits, labels

        return logits, out  # logits: [b,c,k,t] labels: [b,k,t]

    # deprecated
    def forward_streaming(self, indices, base_layer: bool, D_index, mask_proportion: float,
                          return_loss, weighted_loss, q_range,
                          steps_cnt_loss: int,
                          lookahead_frames: int):
        warn('forward_streaming to compute loss is deprecated', DeprecationWarning, stacklevel=2)

        B, K, T = indices.shape
        masked_codes, labels = self.masking(indices, (B, K, T), base_layer=base_layer, mask_proportion=mask_proportion,
                                            q_range=q_range,
                                            frames_mask=steps_cnt_loss,
                                            lookahead_frames=lookahead_frames)  # all fine layer counted in
        masked_codes = rearrange(masked_codes, 'b k t -> b t k')

        if base_layer:
            Q = self.base_layer_n_q
        elif q_range:
            Q = q_range[1]
        else:
            Q = K

        logits, out = self.tokens_to_logits(masked_codes, stage=Q)  # [b,t,k,card]   sum up Q stages embeddings

        # only count loss on last steps
        if lookahead_frames:
            logits = logits[:, -(steps_cnt_loss + lookahead_frames):-lookahead_frames, :Q]
            labels = labels[:, :Q, -(steps_cnt_loss + lookahead_frames):-lookahead_frames]  # b,k,t
        else:
            logits = logits[:, -steps_cnt_loss:, :Q]
            labels = labels[:, :Q, -steps_cnt_loss:]  # b,k,t

        logits = rearrange(logits, 'b n k c -> b c k n')
        if return_loss:
            if weighted_loss:
                loss_bkt = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.ignore_index,
                    reduction='none'
                )
                loss = loss_bkt * self.stage_weight_factor[None, :Q, None]
                loss = loss[labels != self.ignore_index].mean()
            else:
                loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.ignore_index
                )
            return loss, logits, labels
        return logits, out  # logits: [b,c,k,t] labels: [b,k,t]

    def get_coding_order(self, indices_size, context_tensor: torch.Tensor, coarse_first=True, q_ranges=None):
        if coarse_first:
            return self.coding_coder_with_coarse_first(indices_size, context_tensor, q_ranges=q_ranges)
        else:
            return super().get_coding_order(indices_size, context_tensor)

    def coding_coder_with_coarse_first(self, indices_size, context_tensor: torch.Tensor, q_ranges=None):
        """
        update coding order with coarse tokens as stage 0--S-1, and finer tokens from S to 2S-1
        :param q_ranges: finelayer layer range for each group  List[(start_idx, end_idx)]
        :return: [k,t]
        """
        coding_order = super().get_coding_order(indices_size, context_tensor)  # k,t
        slices = context_tensor.max() + 1
        if q_ranges is None:
            coding_order[self.base_layer_n_q:] += slices
        else:  # multi-group fine codes
            for i, q_range in enumerate(q_ranges):
                coding_order[q_range[0]:q_range[1]] += slices * (i + 1)
        return coding_order

    @torch.no_grad()
    def compress_fine_layer(self, rvq_indices, q_ranges: torch.Tensor = None, t_ranges=None, t_unmask=None):
        """
        # K <= base_layer_nq is not valid
        :param t_ranges: 
        :param rvq_indices:
        :return: pdf: [b,k,t,card]  likelihoods: [b,k,t]
        """
        B, K, T = rvq_indices.size()
        assert K > self.base_layer_n_q

        # print(coding_order.size())
        unmask_flag = torch.zeros(K, T, dtype=torch.bool, device=rvq_indices.device)
        unmask_flag[:self.base_layer_n_q] = True  # conditioned on all base_layer tokens

        if t_unmask is not None:
            unmask_flag[:, t_unmask] = True

        masked_codes = torch.where(unmask_flag, rvq_indices, self.mask_token_id)
        masked_codes = rearrange(masked_codes, 'b k t -> b t k')
        logits_btkc, out = self.tokens_to_logits(masked_codes, stage=K)
        logits_bktc = rearrange(logits_btkc, 'b t k c -> b k t c')  # k>=K

        if t_ranges:
            logits_bktc = logits_bktc[:, :K, t_ranges[0]:t_ranges[1]]
            rvq_onehot = F.one_hot(rvq_indices[..., t_ranges[0]:t_ranges[1]], num_classes=self.card)  # [b,K,t,card]
        else:
            logits_bktc = logits_bktc[:, :K]
            rvq_onehot = F.one_hot(rvq_indices, num_classes=self.card)  # [b,K,t,card]

        if t_unmask is not None:  # for non-key slice
            logits_bktc = logits_bktc[:, :, ~t_unmask]

        if q_ranges:
            pdf = logits_bktc[:, q_ranges[0]:q_ranges[1]].softmax(dim=-1)
            likelihoods = (pdf * rvq_onehot[:, q_ranges[0]:q_ranges[1]]).sum(dim=-1)
        else:
            pdf = logits_bktc[:, self.base_layer_n_q:].softmax(dim=-1)  # [b, K - self.base_layer_n_q, T, card]
            likelihoods = (pdf * rvq_onehot[:, self.base_layer_n_q:]).sum(dim=-1)  # [b, K - self.base_layer_n_q, T]

        return likelihoods, pdf

    @torch.no_grad()
    def fixed_context_model(self, rvq_indices, context_tensor, frame_rate: int, keyslice: int = None):
        """
        :param rvq_indices: [B, K, T]
        :param context_tensor: 1 DIM [T_CONTEXT]  or 2 dim [K, T_CONTEXT]
        :return:
        """
        codes_shape = rvq_indices.size()
        coding_order = self.get_coding_coder(codes_shape, context_tensor, coarse_first=False)
        # temporal stride   [0 -- S-1] periodical

        # base layer compression no context model   according to a fixed codebook
        likelihoods_fine, pdf_fine = self.compress_fine_layer(rvq_indices)  # [b,k,t,card]

        if keyslice is not None:
            t_key_frame = coding_order == keyslice
            likelihoods_key = likelihoods_fine[:, :, t_key_frame]
            rate_f_key = - torch.log2(likelihoods_key + 1e-10).sum()
            likelihoods_non_key, pdf_fine = self.compress_fine_layer(rvq_indices, t_unmask=t_key_frame)
            rate_f_nonkey = - torch.log2(likelihoods_non_key + 1e-10).sum()
            rate_f = (rate_f_key + rate_f_nonkey) / (codes_shape[0] * codes_shape[-1]) * frame_rate / 1e3
        else:
            rate_f = - torch.log2(likelihoods_fine + 1e-10).sum() / (
                    codes_shape[0] * codes_shape[-1]) * frame_rate / 1e3
        return {
            # 'accuracy_f': accuracy_f.item(),
            'pdf_fine': pdf_fine,
            'rate_f': rate_f.item()
        }

    @torch.no_grad()
    def streaming_compress_fine_layer(self, rvq_indices, context_tensor, frame_rate: int,
                                      lookahead_frames=0,
                                      interval=1,
                                      max_context_frames: int = 1e6,
                                      real_compress=False):
        B, K, T = rvq_indices.size()
        coding_order = self.coding_coder_with_coarse_first((B, K, T), context_tensor, q_ranges=None)

        likelihoods_fine = torch.zeros(B, K - self.base_layer_n_q, T, dtype=torch.float32, device=rvq_indices.device)
        last_t = 0

        nbits = 0

        for t in range(interval, T + interval, interval):
            start = max(0, t - max_context_frames)
            input_ = rvq_indices[..., start:(t + lookahead_frames)]
            likelihoods_fine[..., last_t:t], pmf = \
                self.compress_fine_layer(input_,
                                         t_ranges=(last_t - start, min(t, T) - start))  # [b,k,t,card]

            if real_compress:
                # encode rvq_indices[:, self.base_layer_n_q:, (last_t - start): (min(t, T) - start)] using ``pmf''  # [b, K - self.base_layer_n_q, T, card]
                pmf_clip = pmf.clamp(1e-10, 1.0).flatten(end_dim=-2)
                symbols = rvq_indices[:, self.base_layer_n_q:, (last_t - start): (min(t, T) - start)].flatten()
                probabilities = pmf_clip.cpu().numpy().astype(np.float64)

                # print("symbol number", symbols.numel(), symbols.size(), probabilities.shape)
                indices = symbols.cpu().numpy().astype(np.int32)
                encoder = constriction.stream.queue.RangeEncoder()
                model_family = constriction.stream.model.Categorical(perfect=False)
                encoder.encode(indices, model_family, probabilities)
                compressed = encoder.get_compressed()

                nbits += encoder.num_bits()
                # real decompress
                decoder = constriction.stream.queue.RangeDecoder(compressed)
                indices_hat = decoder.decode(model_family, probabilities)
                indices_hat = torch.from_numpy(indices_hat).to(rvq_indices.device)

            last_t = t

        rate_f = - torch.log2(likelihoods_fine + 1e-10).sum() / (B * T / frame_rate) / 1000  # kbps
        real_coding_rate_f = nbits / (B * T / frame_rate) / 1000
        # print("rate_f", rate_f, nbits, real_coding_rate_f)
        return {
            'accuracy_f': None,
            'rate_f': rate_f.item(),
            'real_coding_rate_f': real_coding_rate_f
        }

    @torch.no_grad()
    def mask_prediction(self, indices_hat, unmask=None, base_layer=False, stage=None):
        """
        indices_hat  MASKED CODES WITH MASK TOKEN, MASK HAS BEEN APPLIED
        :param base_layer: indicator to predict base layer
        :param rvq_indices: b,k,t  true codes for known pos and mask_token_id for unknown pos
        :return:
        """

        if not base_layer and unmask is not None:
            masked_codes = torch.where(unmask[None], indices_hat, self.mask_token_id)
        else:
            masked_codes = indices_hat
        masked_codes = rearrange(masked_codes, 'b k t -> b t k')
        if not base_layer:
            logits, out = self.tokens_to_logits(masked_codes,
                                                stage=stage or masked_codes.size(-1))
        else:
            logits, out = self.tokens_to_logits(masked_codes,
                                                stage=self.base_layer_n_q)
        return logits, out  # b,t,k,c

    @torch.no_grad()
    def conceal(self,
                rvq_indices,
                coding_order,
                n_slices: int,
                npackets_per_slice: int,
                loss_flag_slices,
                vqlayer_bins: List[int],
                recover_base: bool,
                recover_fine: bool,
                num_stages_to_pred,
                is_entropy_code=True):
        """
        non-streaming concealing
        :param vqlayer_bins:  [2, 2, 4, 8]
        :param rvq_indices:
        :param coding_order: k,t
        :param n_slices:
        :param loss_flag_pt:   (total_n_slices)  total_n_slices =  npackets_per_slice * n_slices
        :param recover_base:
        :param recover_fine:
        :param num_stages_to_pred:
        :return: (codes_hat, valid_code_flag) (all valid if None using quantizer.decode(`) else decode_with_different_stages(`)
        """
        assert vqlayer_bins[0] == self.base_layer_n_q
        B, K, T = rvq_indices.size()
        coding_order_t = coding_order[0]

        loss_flag_ps = loss_flag_slices.reshape(npackets_per_slice, n_slices)
        # map to loss_flag_pt
        unmask_ps = (~loss_flag_ps).cumprod(dim=0).to(torch.bool)  # autoregressive

        loss_flag_ks = torch.repeat_interleave(~unmask_ps, vqlayer_bins, dim=0)[:K]  # todo
        unmask_ks = torch.repeat_interleave(unmask_ps, vqlayer_bins, dim=0)[:K]

        frame_loss_flag = loss_flag_ks[:, coding_order_t]
        unmask_kt = unmask_ks[:, coding_order_t]

        base_t_unmask = unmask_kt[0]
        received_indices = torch.where(frame_loss_flag, self.mask_token_id, rvq_indices)
        indices_hat = received_indices.clone()
        valid_codes_flag = unmask_kt.clone()

        is_base_all_right = torch.all(base_t_unmask)

        if not is_base_all_right:  # base codes lost
            if is_entropy_code:
                valid_codes_flag[self.base_layer_n_q:] = False  # context lost all fine codes unavailable
            if recover_base:  # predict base layer codes
                logits, _ = self.mask_prediction(received_indices, None, base_layer=True)
                logits = rearrange(logits[:, :, :self.base_layer_n_q], 'b n k c -> b c k n')

                indices_hat[:, :self.base_layer_n_q] = torch.where(base_t_unmask,
                                                                   indices_hat[:, :self.base_layer_n_q],
                                                                   logits.argmax(dim=1))  # [b,k,t]
                valid_codes_flag[:self.base_layer_n_q] = True

        elif K > self.base_layer_n_q:  # BASE CODES ALL RECEIVED
            cur_fine_lost = torch.any(loss_flag_ps[1:])
            if cur_fine_lost and recover_fine:
                packets_received = unmask_ps.sum(dim=0)
                npc_r = packets_received.min()
                more_packet_received = packets_received - npc_r
                # num_conditioned_layers = vqlayer_bins[0]
                num_conditioned_layers = vqlayer_bins[:npc_r].sum()
                if num_stages_to_pred is not None:
                    # maxlayers = min(K, num_stages_to_pred + num_conditioned_layers)
                    maxlayers = min(K, num_conditioned_layers + vqlayer_bins[npc_r])
                else:
                    maxlayers = K
                unmask_cond = unmask_kt.clone()
                unmask_cond[maxlayers:] = False

                for i in range(n_slices):
                    if more_packet_received[i]:
                        continue
                    logits, _ = self.mask_prediction(received_indices, unmask_cond, stage=maxlayers)  # [b,t,k,card]
                    logits = rearrange(logits[:, :, num_conditioned_layers: maxlayers], 'b n k c -> b c k n')
                    codes_hat_bkt = logits.argmax(dim=1)  # [b,k,t]

                    t_indicator = coding_order_t == i
                    indices_hat[:, num_conditioned_layers: maxlayers, t_indicator] = \
                        torch.where(frame_loss_flag[None, num_conditioned_layers: maxlayers, t_indicator],
                                    indices_hat[:, num_conditioned_layers: maxlayers, t_indicator],
                                    codes_hat_bkt[..., t_indicator])
                    valid_codes_flag[num_conditioned_layers: maxlayers, t_indicator] = True

        indices_hat = torch.where(indices_hat > self.card,
                                  torch.randint_like(rvq_indices, high=self.card),
                                  indices_hat) 
        codes_hat = indices_hat.transpose(0, 1).contiguous()
        return codes_hat, valid_codes_flag

    # ONLY TWO packets to pack one frame: one for base and one for fine
    @torch.no_grad()
    def streaming_conceal(self, rvq_indices, loss_flag, vqlayer_bins,
                          concealing_mode: str,
                          num_stages_to_pred,
                          lookahead_frames,
                          interval,
                          max_context_frames,
                          compress):
        """
        streaming concealing
        :param rvq_indices:
        :param coding_order:
        :param loss_flag: tensor packet_loss_flag_pt (fec mask applied already) [2,t]  1 indicates loss
        :param base_t_unmask:
        :param recover_base:
        :param recover_fine:
        :param num_stages_to_pred:
        :param lookahead_frames:
        :param interval:
        :param max_context_frames:
        :return:
        """
        assert concealing_mode in ["base", "all", "none"], "Invalid concealing mode"
        recover_base, recover_fine = {"base": (True, False),
                                      "all": (True, True),
                                      "none": (False, False)}[concealing_mode]

        B, K, T = rvq_indices.size()
        received_indices = torch.empty_like(rvq_indices).fill_(self.mask_token_id)  # initialize with mask token

        assert B == 1
        unmask = torch.zeros((K, T), dtype=torch.bool, device=rvq_indices.device)

        assert loss_flag.size(0) == 2
        base_t_unmask = loss_flag[0] == 0  # [T]
        fine_t_unmask = loss_flag[1] == 0
        # initialize codes according to frame-wise loss flag
        received_indices[:, :self.base_layer_n_q, base_t_unmask] = rvq_indices[:, :self.base_layer_n_q,
                                                                   base_t_unmask]
        received_indices[:, self.base_layer_n_q:, fine_t_unmask] = rvq_indices[:, self.base_layer_n_q:,
                                                                   fine_t_unmask]
        unmask[:self.base_layer_n_q, base_t_unmask] = True  # true for not lost
        unmask[self.base_layer_n_q:, fine_t_unmask] = True

        indices_hat = received_indices.clone()
        valid_codes_flag = unmask.clone()
        last_t = 0

        if K <= self.base_layer_n_q:
            # only base layer codes
            for t in range(interval, T + interval, interval):
                # last_t to t
                start = max(t - max_context_frames, 0)
                cur_base_lost = torch.any(loss_flag[0, last_t:t])

                if cur_base_lost:
                    if recover_base:
                        # recover base codes
                        logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)], unmask,
                                                         base_layer=True)
                        # not using indices_hat (prevent error propagation, align with training)
                        logits = rearrange(logits[:, last_t - start:min(t, T) - start, :self.base_layer_n_q],
                                           'b n k c -> b c k n')
                        indices_hat[:, :self.base_layer_n_q, last_t:t] = torch.where(
                            base_t_unmask[None, None, last_t:t],
                            indices_hat[:, :self.base_layer_n_q, last_t:t],
                            logits.argmax(dim=1))  # [b,k,t]  # only update positions which are lost
                        valid_codes_flag[:self.base_layer_n_q, last_t:t] = True
                last_t = t

        else:  # fine codes available
            for t in range(interval, T + interval, interval):
                # last_t to t
                start = max(t - max_context_frames, 0)
                cur_base_lost = torch.any(loss_flag[0, last_t:t])
                context_lost = compress and torch.any(loss_flag[0, start:(t + lookahead_frames)])  # contextual base codes
                cur_fine_lost = context_lost or torch.any(loss_flag[1, last_t:t])  # context

                if context_lost:
                    valid_codes_flag[self.base_layer_n_q:, last_t:t] = False  # deactivate fine codes
                    unmask[self.base_layer_n_q:, last_t:t] = False

                if cur_base_lost:
                    if recover_base:
                        # recover base codes
                        logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)], unmask,
                                                         base_layer=True)
                        logits = rearrange(logits[:, last_t - start:min(t, T) - start, :self.base_layer_n_q],
                                           'b n k c -> b c k n')
                        indices_hat[:, :self.base_layer_n_q, last_t:t] = torch.where(
                            base_t_unmask[None, None, last_t:t],
                            indices_hat[:, :self.base_layer_n_q, last_t:t],
                            logits.argmax(dim=1))  # [b,k,t]  # only update positions which are lost
                        valid_codes_flag[:self.base_layer_n_q, last_t:t] = True

                else:  # cur base received
                    if cur_fine_lost:
                        if not context_lost and recover_fine:
                            # contextual base codes all received, but cur fine codes lost
                            # print("recover fine")
                            if num_stages_to_pred:
                                maxlayers = min(K, num_stages_to_pred + self.base_layer_n_q)
                                logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)],
                                                                 unmask[..., start:(t + lookahead_frames)],
                                                                 stage=maxlayers)  # [b,t,k,card]

                                logits = rearrange(logits[:, last_t - start:min(t, T) - start,
                                                   self.base_layer_n_q: maxlayers],
                                                   'b n k c -> b c k n')
                                codes_hat_bkt = logits.argmax(dim=1)

                                indices_hat[:, self.base_layer_n_q: maxlayers, last_t:t] = \
                                    torch.where(
                                        fine_t_unmask[None, None, last_t:t],
                                        indices_hat[:, self.base_layer_n_q: maxlayers, last_t:t],
                                        codes_hat_bkt)

                                valid_codes_flag[self.base_layer_n_q: maxlayers, last_t:t] = True
                            else:
                                raise NotImplementedError("all stage prediction")
                last_t = t

        indices_hat = torch.where(indices_hat > self.card,
                                  torch.randint_like(rvq_indices, high=self.card),
                                  indices_hat)
        codes_hat = indices_hat.transpose(0, 1).contiguous()
        return codes_hat, valid_codes_flag

    # MULTIPLE packets: one for base and others for fine, single bandwidth
    @torch.no_grad()
    def streaming_conceal_progressive(self, rvq_indices, loss_flag_pt, vqlayer_bins,
                                      concealing_mode: str,
                                      num_stages_to_pred: int,
                                      lookahead_frames: int,
                                      interval: int,
                                      max_context_frames: int,
                                      compress: bool):
        """
        :param vqlayer_bins:
        :param rvq_indices:
        :param loss_flag_pt: tensor packet_loss_flag_pt (fec mask applied already) [k,t]  1 indicates loss
        :param concealing_mode: str ['base', 'all', 'None']
        :param num_stages_to_pred:
        :param lookahead_frames:
        :param interval:
        :param max_context_frames:
        :param compress: whether to entropy modeling at the transmitter
        :return:
        """

        if concealing_mode == 'base':
            recover_base, recover_fine = True, False
        elif concealing_mode == 'all':
            recover_base, recover_fine = True, True
        elif concealing_mode == 'None':
            recover_base, recover_fine = False, False
        else:
            raise NotImplementedError("Invalid concealing mode")

        assert vqlayer_bins[0] == self.base_layer_n_q
        B, K, T = rvq_indices.size()
        assert B == 1

        npackets = loss_flag_pt.size(0)
        layer_bins = torch.as_tensor(vqlayer_bins[:npackets], dtype=torch.int, device=rvq_indices.device)

        frame_loss_flag = torch.repeat_interleave(loss_flag_pt, layer_bins, dim=0)[:K]  # [k,t]

        received_indices = torch.where(frame_loss_flag, self.mask_token_id, rvq_indices)
        indices_hat = received_indices.clone()

        unmask_pkts = (~loss_flag_pt).cumprod(dim=0).to(torch.bool)  # rvq autoregressive
        unmask_frms = torch.repeat_interleave(unmask_pkts, layer_bins, dim=0)[:K]  # [k,t]

        valid_codes_flag = unmask_frms.clone()
        last_t = 0

        if K <= self.base_layer_n_q:  # base layer codes available only
            for t in range(interval, T + interval, interval):
                # last_t to t
                start = max(t - max_context_frames, 0)
                cur_base_lost = torch.any(loss_flag_pt[0, last_t:t])

                if cur_base_lost:
                    if recover_base:  # recover base codes
                        logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)], None,
                                                         base_layer=True)  # not using indices_hat
                        logits = rearrange(logits[:, last_t - start:min(t, T) - start, :self.base_layer_n_q],
                                           'b n k c -> b c k n')
                        indices_hat[:, :self.base_layer_n_q, last_t:t] = torch.where(
                            loss_flag_pt[None, 0, last_t:t],
                            logits.argmax(dim=1),
                            indices_hat[:, :self.base_layer_n_q, last_t:t]
                        )  # [b,k,t]  # only update positions which are lost
                        valid_codes_flag[:self.base_layer_n_q, last_t:t] = True
                last_t = t

        else:  # fine codes transmitted
            for t in range(interval, T + interval, interval):
                start = max(t - max_context_frames, 0)
                cur_base_lost = torch.any(loss_flag_pt[0, last_t:t])

                context_lost = compress and torch.any(
                    loss_flag_pt[0, start:(t + lookahead_frames)])  # contextual base codes
                cur_fine_lost = context_lost or torch.any(loss_flag_pt[1:, last_t:t])

                if context_lost:  # compress: contextual modelling available at the transmitter
                    valid_codes_flag[self.base_layer_n_q:, last_t:t] = False  # deactivate fine codes
                    unmask_frms[self.base_layer_n_q:, last_t:t] = False  # context(pdf) lost

                if cur_base_lost:
                    valid_codes_flag[self.base_layer_n_q:, last_t:t] = False
                    unmask_frms[self.base_layer_n_q:, last_t:t] = False
                    if recover_base:
                        # recover base codes
                        logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)], None,
                                                         base_layer=True)
                        logits = rearrange(logits[:, last_t - start:min(t, T) - start, :self.base_layer_n_q],
                                           'b n k c -> b c k n')
                        indices_hat[:, :self.base_layer_n_q, last_t:t] = torch.where(
                            loss_flag_pt[None, 0, last_t:t],
                            logits.argmax(dim=1),
                            indices_hat[:, :self.base_layer_n_q, last_t:t],
                        )  # [b,k,t]  # only update positions which are lost
                        valid_codes_flag[:self.base_layer_n_q, last_t:t] = True

                else:  # cur base received, recover fine codes
                    if cur_fine_lost and recover_fine:
                        if not context_lost:  # able to recover
                            # contextual base codes all received, but cur fine codes lost
                            # find out from which layer to recover
                            num_conditioned_packets = unmask_pkts[:, last_t:t].all(dim=-1).sum()
                            num_conditioned_layers = layer_bins[:num_conditioned_packets].sum()

                            if num_stages_to_pred:
                                maxlayers = min(K, num_stages_to_pred + num_conditioned_layers)
                                logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)],
                                                                 unmask_frms[..., start:(t + lookahead_frames)],
                                                                 stage=maxlayers)  # [b,t,k,card]

                                logits = rearrange(logits[:, last_t - start:min(t, T) - start,
                                                   num_conditioned_layers: maxlayers],
                                                   'b n k c -> b c k n')
                                codes_hat_bkt = logits.argmax(dim=1)

                                indices_hat[:, num_conditioned_layers: maxlayers, last_t:t] = \
                                    torch.where(
                                        frame_loss_flag[None, num_conditioned_layers: maxlayers, last_t:t],
                                        codes_hat_bkt,
                                        indices_hat[:, num_conditioned_layers: maxlayers, last_t:t])

                                valid_codes_flag[num_conditioned_layers: maxlayers, last_t:t] = True
                            else:
                                raise NotImplementedError("all stage prediction not implemented")

                last_t = t

        indices_hat = torch.where(indices_hat > self.card,
                                  torch.randint_like(rvq_indices, high=self.card),
                                  indices_hat)
        codes_hat = indices_hat.transpose(0, 1).contiguous()
        return codes_hat, valid_codes_flag


    # MULTIPLE packets: one for base and others for fine, variable bandwidth
    @torch.no_grad()
    def streaming_conceal_progressive_variable_bw(self, rvq_indices, seg_layers, layer2packet, seg_frames,
                                                  loss_flag_pts, vqlayer_bins,
                                                  recover_base: bool, recover_fine: bool, num_stages_to_pred,
                                                  lookahead_frames, interval, max_context_frames,
                                                  compress):
        """
        :param vqlayer_bins:
        :param rvq_indices:
        :param coding_order:
        :param loss_flag_pt: tensor packet_loss_flag_pt (fec mask applied already) [k,t]  1 indicates loss
        :param base_t_unmask:
        :param recover_base:
        :param recover_fine:
        :param num_stages_to_pred:
        :param lookahead_frames:
        :param interval:
        :param max_context_frames:
        :return:
        """
        assert vqlayer_bins[0] == self.base_layer_n_q
        B, K, T = rvq_indices.size()
        assert B == 1

        maxpackets = layer2packet[K]
        layer_bins = torch.as_tensor(vqlayer_bins[:maxpackets], dtype=torch.int, device=rvq_indices.device)

        def get_phase_pkt_flag(phase, n_layer):
            loss_flag_pt = torch.as_tensor(loss_flag_pts[phase],
                                           device=rvq_indices.device)  # [maxpackets, T]  1 for loss, 0 for good

            unmask_pkts = (1 - loss_flag_pt).cumprod(dim=0).to(torch.bool)  # autoregressive
            npc = layer2packet[n_layer]
            unmask_pkts[npc:] = False

            loss_flag_pt[npc:] = -1  # -1: no transmission
            return loss_flag_pt, unmask_pkts

        loss_flag_pt, unmask_pkts = [], []
        for phase, n_layer in enumerate(seg_layers):
            loss_flag_pt_, unmask_pkts_ = get_phase_pkt_flag(phase, n_layer)
            loss_flag_pt.append(loss_flag_pt_)
            unmask_pkts.append(unmask_pkts_)

        loss_flag_pt = torch.cat(loss_flag_pt, dim=-1)  # 0 for good, 1 for lost, -1 for no transmission
        unmask_pkts = torch.cat(unmask_pkts, dim=-1)

        frame_loss_flag = torch.repeat_interleave(loss_flag_pt, layer_bins, dim=0)[:K]
        unmask_frms = torch.repeat_interleave(unmask_pkts, layer_bins, dim=0)[:K]  # [k,t]

        received_indices = torch.where(frame_loss_flag.ne(0), self.mask_token_id, rvq_indices)  # -1: no transmission
        indices_hat = received_indices.clone()
        valid_codes_flag = unmask_frms.clone()

        last_t = 0

        tip_points = np.cumsum(seg_frames)
        phase = 0

        for t in range(interval, T + interval, interval):  # last_t to t
            if t > tip_points[phase]:
                phase += 1

            start = max(t - max_context_frames, 0)
            cur_base_lost = torch.any(loss_flag_pt[0, last_t:t])

            if K <= self.base_layer_n_q:  # only base layer codes
                if cur_base_lost:
                    if recover_base:  # recover base codes
                        logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)], None,
                                                         base_layer=True)  # DO not use indices_hat
                        logits = rearrange(logits[:, last_t - start:min(t, T) - start, :self.base_layer_n_q],
                                           'b n k c -> b c k n')
                        indices_hat[:, :self.base_layer_n_q, last_t:t] = torch.where(
                            loss_flag_pt[None, 0, last_t:t].eq(1),
                            logits.argmax(dim=1),
                            indices_hat[:, :self.base_layer_n_q, last_t:t]
                        )  # [b,k,t]  # only update positions which are lost
                        valid_codes_flag[:self.base_layer_n_q, last_t:t] = True
            else:  # fine codes available
                context_lost = torch.any(loss_flag_pt[0, start:(t + lookahead_frames)])  # contextual base codes
                cur_fine_lost = context_lost or torch.any(loss_flag_pt[1:, last_t:t])  # context
                if context_lost:
                    valid_codes_flag[self.base_layer_n_q:, last_t:t] = False  # deactivate fine codes
                    unmask_frms[self.base_layer_n_q:, last_t:t] = False  #context

                if compress:
                    # compress fine codes # entropy coding (optional)
                    _, pdf_fine = self.compress_fine_layer(rvq_indices[..., start:(t + lookahead_frames)],
                                                           q_ranges=None,
                                                           t_ranges=(last_t - start, min(t, T) - start))

                if cur_base_lost:
                    if recover_base:
                        # recover base codes
                        logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)],
                                                         None,
                                                         base_layer=True)
                        logits = rearrange(logits[:, last_t - start:min(t, T) - start, :self.base_layer_n_q],
                                           'b n k c -> b c k n')
                        indices_hat[:, :self.base_layer_n_q, last_t:t] = torch.where(
                            loss_flag_pt[None, 0, last_t:t].eq(1),
                            logits.argmax(dim=1),
                            indices_hat[:, :self.base_layer_n_q, last_t:t],
                        )  # [b,k,t]  # only update positions which are lost
                        valid_codes_flag[:self.base_layer_n_q, last_t:t] = True
                    else:
                        # w.o. prediction random indices for lost base codes
                        pass
                else:  # cur base received, consider decompressing / recovering fine codes
                    if cur_fine_lost and recover_fine:
                        if not context_lost:  # able to recover
                            # contextual base codes all received, but cur fine codes lost
                            # print("recover fine")
                            # find from which layer to recover
                            num_conditioned_packets = unmask_pkts[:, last_t:t].all(dim=-1).sum()
                            num_conditioned_layers = layer_bins[:num_conditioned_packets].sum()

                            if num_stages_to_pred:
                                maxlayers = min(K, num_stages_to_pred + num_conditioned_layers)
                                logits, _ = self.mask_prediction(received_indices[..., start:(t + lookahead_frames)],
                                                                 unmask_frms[..., start:(t + lookahead_frames)],
                                                                 stage=maxlayers)  # [b,t,k,card]

                                logits = rearrange(logits[:, last_t - start:min(t, T) - start,
                                                   num_conditioned_layers: maxlayers],
                                                   'b n k c -> b c k n')
                                codes_hat_bkt = logits.argmax(dim=1)
                                indices_hat[:, num_conditioned_layers: maxlayers, last_t:t] = \
                                    torch.where(
                                        frame_loss_flag[None, num_conditioned_layers: maxlayers, last_t:t].eq(0),
                                        indices_hat[:, num_conditioned_layers: maxlayers, last_t:t],
                                        codes_hat_bkt
                                    )
                                valid_codes_flag[num_conditioned_layers: maxlayers, last_t:t] = True
                            else:
                                raise NotImplementedError("all stage prediction")
                    elif compress:  # retrieve fine codes when packet for fine codes is received
                        _, pdf = self.compress_fine_layer(indices_hat[..., start:(t + lookahead_frames)],
                                                          t_ranges=(last_t - start, min(t, T) - start))
                        # retrieve codes from pdf  entropy decoding
            last_t = t

        indices_hat = torch.where(indices_hat > self.card,
                                  torch.randint_like(rvq_indices, high=self.card),
                                  indices_hat)
        codes_hat = indices_hat.transpose(0, 1).contiguous()

        return codes_hat, valid_codes_flag, frame_loss_flag
