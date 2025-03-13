import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_RVQ import Encodec_RVQ
from entropy_model import LMConformer
import math
import random
import numpy as np
import time
from einops import rearrange
from itertools import product
from typing import List
import torchaudio
import typing as tp

lm_list = {
    "LMConformer": LMConformer,
}


class model_RVQ_LM_base(nn.Module):
    def __init__(self, lm_name: str, config, lm_config: dict = None):
        super().__init__()
        assert lm_name in lm_list, "illegal lm_name"

        encodec_config = dict(dimension=config.dimension,
                              pretrained=config.load,
                              ckpt_path=config.Encodec_Path)
        self.encodec = Encodec_RVQ.causal_model_18khz(config, **encodec_config)

        bw_per_q = math.log2(config.quantizer_bins) * self.frame_rate
        nq_options = [math.floor(bw * 1000 / bw_per_q) for bw in config.target_bandwidths]
        max_nq = int(max(1, max(nq_options)))
        self.bw_per_q = bw_per_q
        self.max_nq = max_nq

        if lm_config is None:
            lm_config = dict(dim=encodec_config['dimension'],
                             n_q=max_nq,
                             card=config.quantizer_bins)
        else:
            lm_config['n_q'] = max_nq  # modify n_q

        print('LM config bw_per_q nq_options:', bw_per_q, nq_options)
        # if lm_name == 'LMConformer2_g':
        lm_config['n_q_options'] = nq_options

        self.LM = lm_list[lm_name](**lm_config)
        self.codebook_num = lm_config['n_q']
        self.codebook_dim = lm_config['dim']
        self.codebook_card = lm_config['card']

        self.CELoss = torch.nn.CrossEntropyLoss(reduction='none')
        self.config = config

        stage_weight_factor = np.linspace(5.0, 1.0, self.max_nq)

        self.register_buffer('stage_weight_factor', torch.as_tensor(stage_weight_factor))

    @property
    def frame_rate(self):
        return self.encodec.frame_rate

    @property
    def bandwidth(self):
        return self.encodec.bandwidth

    @property
    def target_bandwidths(self):
        # return self.encodec.target_bandwidths
        return self.config.target_bandwidths

    @property
    def num_target_bandwidths(self):
        return len(self.target_bandwidths)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Base class method not defined')

    @torch.no_grad()
    def inference(self, *args, **kwargs):
        raise NotImplementedError('Base class method not defined')

    @torch.no_grad()
    def streaming_inference(self):
        pass

    def count_parameters(self):
        r"""Count parameters of encoders"""
        print(f'Model params = {sum([p.numel() for p in self.parameters()]) / 1e6 :.3f} M')

    def count_LM_parameters(self):
        r"""Count parameters of encoders"""
        print(f'LM params = {sum([p.numel() for p in self.LM.parameters()]) / 1e6 :.3f} M')

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.encodec.set_target_bandwidth(bandwidth)

    def count_loss(self, mode, logits_bckt, rvq_indices_bkt, mask, weighted: bool):
        """

        :param mode:
        :param logit_bckt:
        :param rvq_indices_bckt:
        :param mask: b,k,t   False position to pred
        :return:
        """
        ce_loss = self.CELoss(logits_bckt, rvq_indices_bkt)  # [b,k,t]
        if mode == 'all_pos':
            ce_loss = ce_loss.mean()
        elif mode == 'mask_pos':
            if weighted:
                K = rvq_indices_bkt.size(1)
                factor = self.stage_weight_factor[None, :K, None]
                ce_loss = ce_loss * factor
                mask[:, ~self.stage_weight_factor[:K].to(bool), :] = True

            ce_loss = ce_loss[mask == 0].mean()

        return ce_loss

    def eval_pred(self, mode, codes_hat_bkt, rvq_indices_bkt, pred_mask, weighted: bool):
        if mode == 'all_pos':
            n_right = torch.sum(codes_hat_bkt == rvq_indices_bkt)
            n_total = rvq_indices_bkt.numel()
        else:  # mode == 'mask_pos'
            if weighted:
                K = codes_hat_bkt.size(1)
                pred_mask[:, ~self.stage_weight_factor[:K].to(bool), :] = True
            n_right = torch.sum((codes_hat_bkt == rvq_indices_bkt) * (pred_mask == 0))
            n_total = (pred_mask == 0).sum()

        accuracy = n_right / n_total
        return n_right, n_total, pred_mask, accuracy

    def fixed_context_model(self, rvq_indices, context_tensor, weighted: bool):
        """
        :param rvq_indices: [B, K, T]
        :param context_tensor: 1 DIM [T_CONTEXT]  or 2 dim [K, T_CONTEXT]
        :return:
        """
        codes_shape = rvq_indices.size()
        coding_order = self.LM.get_coding_order(codes_shape, context_tensor)  # [k,t]
        # print('coding_order', context_tensor, rvq_indices.size(), coding_order.size())
        rvq_onehot = F.one_hot(rvq_indices, num_classes=self.codebook_card)  # [b,k,t,card]
        n_steps = context_tensor.max() + 1
        pdf = torch.zeros_like(rvq_onehot, dtype=torch.float32)
        for step in range(n_steps):
            ctx_idx = (coding_order < step)
            to_encode_idx = (coding_order == step)  # [k,t]
            if step == 0:
                input_tokens = torch.empty_like(rvq_indices).fill_(self.LM.mask_token_id)
            else:
                input_tokens[:, to_encode_idx] = rvq_indices[:, to_encode_idx] + self.LM.num_special_tokens

            logits = self.LM.inference(input_tokens, ctx_idx)  # [b,k,t,card]
            pdf[:, to_encode_idx, :] = logits[:, to_encode_idx, :].softmax(dim=-1)

        likelihoods = (pdf * rvq_onehot).sum(dim=-1)  # [b,k,t]

        # eval pred  TODO
        if weighted:
            stage_flag = self.stage_weight_factor[:codes_shape[1]] > 0
            stage_indices_hat = pdf.argmax(-1)[:, stage_flag, :]
            target_stage_indices = rvq_indices[:, stage_flag, :]
            accuracy = (stage_indices_hat == target_stage_indices).sum() / target_stage_indices.numel()
        else:
            accuracy = (pdf.argmax(-1) == rvq_indices).sum() / rvq_indices.numel()
        return likelihoods, accuracy

    def get_context_tensor(self, context_mode: str, device, **kwargs):
        """
        pattern , to define the coding order within one slice
        :param context_mode:
        :param device:
        :param kwargs:  'slices'
        :return:
        """
        if context_mode == 'temporal_stride':
            # stride == slices
            context_tensor = torch.arange(kwargs['slices'], device=device)  # stride == T
        elif context_mode == 'stage_autoregressive':
            K = kwargs['n_q']
            context_tensor = torch.arange(K, device=device)[:, None]  # [k,1]
        elif context_mode == 'streaming':
            assert kwargs['slices'] <= 6  # latency
            context_tensor = None  # todo
            pass
        else:
            raise NotImplementedError('Invalid context mode')
        return context_tensor


class SoundSpring_speech16k(model_RVQ_LM_base):
    def __init__(self, lm_name: str, config, lm_config, streaming=False):
        super(SoundSpring_speech16k, self).__init__(lm_name, config, lm_config)
        assert isinstance(self.LM, LMConformer)
        self.count_parameters()
        self.count_LM_parameters()
        self.streaming = streaming

        self.baselayer_entropy = np.sum(config.rvq_base_entropy_H1)  # order-1 entropy
        self.rate_b = self.baselayer_entropy * self.frame_rate / 1000


    def forward(self, x, D_index,
                update_D: bool = False, update_G: bool = False, lm_only=False, weighted=False,
                steps_cnt_loss: int = 5,
                lookahead_frames: int = 3):
        bandwidth = self.target_bandwidths[D_index]
        with torch.no_grad():
            if self.encodec.normalize:
                mono = x.mean(dim=1, keepdim=True)
                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                scale = 1e-8 + volume
                x = x / scale
                scale = scale.view(-1, 1)
            else:
                scale = None

            emb = self.encodec.encoder(x)
            rvq_indices = self.encodec.quantizer.encode(emb, self.frame_rate, bandwidth)
            rvq_indices = rvq_indices.detach().transpose(0, 1).contiguous()

        loss_base_layer, logits_base_layer, labels_base_layer = self.LM(self.streaming, rvq_indices,
                                                                        D_index=D_index, base_layer=True,
                                                                        weighted_loss=weighted,
                                                                        steps_cnt_loss=steps_cnt_loss,
                                                                        lookahead_frames=lookahead_frames
                                                                        )
        if self.LM.base_layer_n_q * self.bw_per_q < bandwidth * 1000:
            loss_fine_layer, logits_fine_layer, labels_fine_layer = self.LM(self.streaming, rvq_indices,
                                                                            D_index=D_index, base_layer=False,
                                                                            weighted_loss=weighted,
                                                                            steps_cnt_loss=steps_cnt_loss,
                                                                            lookahead_frames=lookahead_frames
                                                                            )
            # print(labels_base_layer.size(), logits_base_layer.size(), labels_fine_layer.size(), logits_fine_layer.size())
            return {
                'base': (loss_base_layer, logits_base_layer, labels_base_layer),
                'fine': (loss_fine_layer, logits_fine_layer, labels_fine_layer)
            }
        else:
            return {
                'base': (loss_base_layer, logits_base_layer, labels_base_layer),
                'fine': None
            }

    @torch.no_grad()
    def compute_base_layer_stat(self, x):
        if self.encodec.normalize:  # false
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encodec.encoder(x)
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.target_bandwidths[0])  # [k,b,t]

        return codes

    @torch.no_grad()
    def inference(self, x, frame_loss_rate=0.05, context_mode='temporal_stride', slices=2) -> dict:
        """
        :param x:
        :param frame_loss_rate:
        :param context_mode:
        :return:
        """
        emb = self.encodec.encoder(x)  # [B,dimension,T]
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.bandwidth)  # [n_q, B, T]

        rvq_indices = codes.detach().transpose(0, 1).contiguous()  # [B, n_q, T]
        # rvq_onehot_indices = F.one_hot(rvq_indices, num_classes=self.codebook_card)  #[B, K, T, card]

        # entropy encoding
        results = self.LM.fixed_context_model(rvq_indices,
                                              self.get_context_tensor(context_mode,
                                                                      x.device,
                                                                      slices=slices),
                                              frame_rate=self.frame_rate)

        pmf_finetokens = results['pdf_fine']

        results['rate_b'] = self.rate_b
        results['rate_kbps_no_fec'] = self.rate_b + results['rate_f']
        results['pmf_finetokens'] = pmf_finetokens
        return results

    @torch.no_grad()
    def streaming_inference(self, x, frame_loss_rate=0.05,
                            context_mode='temporal_stride', slices=2,
                            lookahead_frames: int = 2, interval: int = 2, max_context_frames: int = None,
                            keyframe_detect_config=None):
        """
        usage in test.py test_compression()
        :param x:
        :param frame_loss_rate:
        :param context_mode:
        :param slices:
        :param lookahead_frames:
        :param interval:  
        :param max_context_frames:   context span  lead to
        :return:
        """

        start_time = time.time()  
        emb = self.encodec.encoder(x)  # [B,dimension,T]
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        enc_duration = time.time() - start_time

        rvq_indices = codes.detach().transpose(0, 1).contiguous()

        if keyframe_detect_config is not None:
            keyframe_flag = self.vad_detector(x, **keyframe_detect_config)

        results = self.LM.streaming_compress_fine_layer(rvq_indices,
                                                        self.get_context_tensor(context_mode,
                                                                                x.device, slices=slices),
                                                        frame_rate=self.frame_rate,
                                                        lookahead_frames=lookahead_frames,
                                                        interval=interval,
                                                        max_context_frames=max_context_frames)
        duration = time.time() - start_time
        rtf_enc = enc_duration / rvq_indices.size(-1) * self.frame_rate
        rtf_enc_ec = duration / rvq_indices.size(-1) * self.frame_rate

        # print(f'encoding rtf: {rtf_enc:.2f} {rtf_enc_ec:.2f}')
        results['rate_b'] = self.rate_b
        results['rate_kbps_no_fec'] = self.rate_b + results['rate_f']
        results['rtf_enc'] = rtf_enc
        results['rtf_enc_ec'] = rtf_enc_ec
        return results

    @DeprecationWarning
    def simulate_concealing(self, x, context_mode: str, n_slices: int, packet_loss: float,
                             concealing_mode:str = 'base',
                             num_stages_to_pred: int = None,
                             fec=True, fec_mode='101',
                             lookahead_frames: int = 2, interval: int = 2, max_context_frames: int = None):
        """
        WITHOUT ENTROPY MODELING

        :param recover_fine:
        :param x:
        :param context_mode:
        :param n_slices:
        :param packet_loss:
        :param recover_base: -1 denotes w.o. concealment
        :param num_stages_to_pred:  None: predict all lost fine tokens
        :param fec:
        :return:
        """
        assert x.size(0) == 1  # B=1
        context_tensor = self.get_context_tensor(context_mode, x.device, slices=n_slices)

        if self.encodec.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encodec.encoder(x)  # [B,dimension,T]
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.bandwidth)  # [n_q, B, T]

        if packet_loss == 0:
            embed_hat = self.encodec.quantizer.decode(codes)  # [B,dimension,T]
            x_hat = self.encodec.decoder(embed_hat)
            x_hat = x_hat[:, :, :x.size(-1)]
            if scale is not None:
                x_hat = x_hat * scale.view(-1, 1, 1)
        elif packet_loss >= 1: # all packets lost
            raise ValueError("Packet loss cannot exceed 100%")
        else:  # recover using MLM
            rvq_indices = codes.detach().transpose(0, 1).contiguous()  # [B, k, T]
            B, K, T = rvq_indices.size()
            coding_order = self.LM.get_coding_order((B, K, T), context_tensor, coarse_first=True)  # K,T
            total_n_slices = coding_order.max() + 1
            base_slices = n_slices
            # simulate_loss
            # n_slices
            loss_flag = np.zeros(total_n_slices)  # 1 for loss
            loss_flag[:math.floor(total_n_slices * packet_loss)] = 1
            # loss_flag = np.random.permutation(loss_flag)

            np.random.shuffle(loss_flag)
        

            packet_flag_pt = self.packet_flag_pt(T, npackets_per_frame, packet_loss_trace,
                                             interval, lookahead_frames,
                                             fec, fec_mode, variable=False)

            # get received masked indices tensor
            codes_hat, valid_codes_flag = conceal_func(rvq_indices,
                                                   packet_flag_pt,
                                                   vqlayer_bins,
                                                   concealing_mode,
                                                   num_stages_to_pred,
                                                   lookahead_frames=lookahead_frames,
                                                   interval=interval,
                                                   max_context_frames=max_context_frames or 50,
                                                   compress=compress)
            if valid_codes_flag is None:
                embed_hat = self.encodec.quantizer.decode(codes_hat)
            else:
                embed_hat = self.encodec.quantizer.decode_with_different_stages(codes_hat,
                                                                                valid_q_bkt_mask=valid_codes_flag.tile(
                                                                                    B, 1, 1))

            x_hat = self.encodec.decoder(embed_hat)
            x_hat = x_hat[:, :, :x.size(-1)]


            if scale is not None:
                x_hat = x_hat * scale.view(-1, 1, 1)

        return {
            'x_hat': x_hat,
        }

    @torch.no_grad()
    def simulate_packet_loss_example(self, x, context_mode: str, n_slices: int, packet_loss: float,
                                     num_stages_to_pred: int = None, fec=True, fec_mode='101',
                                     tracer_generator=None, sim_mode="ratio",
                                     compress:bool=False):
        assert x.size(0) == 1  # B=1
        context_tensor = self.get_context_tensor(context_mode, x.device, slices=n_slices)

        start_time = time.time()
        if self.encodec.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encodec.encoder(x)  # [B,dimension,T]
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.bandwidth)  # [n_q, B, T]

        rvq_indices = codes.detach().transpose(0, 1).contiguous()  # [B, k, T]

        if compress:
            results = self.LM.fixed_context_model(rvq_indices,
                                                  self.get_context_tensor(context_mode,
                                                                          x.device,
                                                                          slices=n_slices),
                                                  frame_rate=self.frame_rate)
            duration = time.time() - start_time
            rtf = duration / rvq_indices.size(-1) * self.frame_rate
            # print('encoding rtf:', rtf)

            results['rate_b'] = self.rate_b
            results['rate_kbps_no_fec'] = self.rate_b + results['rate_f']
            print('compression result', results['rate_b'], results['rate_kbps_no_fec'])

        B, K, T = rvq_indices.size()
        coding_order = self.LM.get_coding_order((B, K, T), context_tensor, coarse_first=True)  # K,T
        total_n_slices = coding_order.max() + 1
        base_slices = n_slices

        # simulate_loss
        # n_slices
        if tracer_generator:
            loss_flag = tracer_generator.generate(T)
            pass  # todo
        elif sim_mode == 'ratio':
            loss_flag = np.zeros(total_n_slices)  # 1 for loss
            loss_flag[:math.floor(total_n_slices * packet_loss)] = 1
            loss_flag = np.random.permutation(loss_flag)
        else:  # i.i.d binary bernoulli(packet_loss)
            loss_flag = np.random.binomial(1, packet_loss, (total_n_slices))

        # indices_hat0 = torch.empty_like(rvq_indices).fill_(self.LM.mask_token_id)  # initialize with mask
        # mask = torch.zeros_like(co
        # ding_order, dtype=torch.bool)
        # initialize mask according to loss_flag
        base_t_unmask = torch.zeros_like(coding_order[0], dtype=torch.bool)

        for i in range(base_slices):
            # w. fec
            if loss_flag[i] == 1:  # this packet is lost
                if not fec: continue
                if fec_mode == '101':
                    fec_flag = loss_flag[(i - 1 + n_slices) % n_slices] == 0 or loss_flag[(i + 1) % n_slices] == 0
                elif fec_mode == '01':
                    fec_flag = loss_flag[(i + 1 + n_slices) % n_slices] == 0
                elif fec_mode == '10':
                    fec_flag = loss_flag[(i - 1 + n_slices) % n_slices] == 0
                elif fec_mode == '110':
                    fec_flag = loss_flag[(i - 1 + n_slices) % n_slices] == 0 or loss_flag[
                        (i - 2 + n_slices) % n_slices] == 0
                else:
                    raise Exception('invalid fec mode')
                if not fec_flag:
                    # print("base layer lost")
                    # is_base_all_right = False
                    continue
            base_t_unmask[coding_order[0, :] == i] = True

        x_hat_list = {}
        codes_hat_list = {}
        valid_codes_flag_list = {}
        for recover_base, recover_fine in [(-1, False), (0, False), (0, True)]:
            # get received masked indices tensor
            codes_hat, valid_codes_flag = self.LM.conceal(rvq_indices, coding_order, n_slices, loss_flag, base_t_unmask,
                                                          recover_base, recover_fine, num_stages_to_pred)

            if valid_codes_flag is None:
                embed_hat = self.encodec.quantizer.decode(codes_hat)
            else:
                embed_hat = self.encodec.quantizer.decode_with_different_stages(codes_hat,
                                                                                valid_q_bkt_mask=valid_codes_flag.tile(
                                                                                    B, 1, 1))

            x_hat = self.encodec.decoder(embed_hat)
            x_hat = x_hat[:, :, :x.size(-1)]

            if scale is not None:
                x_hat = x_hat * scale.view(-1, 1, 1)

            tag_base = '_base' if recover_base != -1 else ''
            tag_fine = '_fine' if recover_fine else ''
            x_hat_list['x_hat' + tag_base + tag_fine] = x_hat
            codes_hat_list['x_hat' + tag_base + tag_fine] = codes_hat
            valid_codes_flag_list['x_hat' + tag_base + tag_fine] = valid_codes_flag

        return x_hat_list, codes_hat_list, valid_codes_flag_list
        # return x_hat_list, codes_hat_list, valid_codes_flag_list, is_base_all_right

    @torch.no_grad()
    def simulate_streaming_example(self, x, context_mode: str, n_slices: int, packet_loss: float, vqlayer_bins=None,
                                   num_stages_to_pred: int = None, fec=True, fec_mode='01',
                                   trace_generator=None,
                                   sim_mode='ratio', compress=False,
                                   concealing_mode:str = 'base',
                                   lookahead_frames: int = 2, interval: int = 2, max_context_frames: int = None):
        assert x.size(0) == 1  # B=1
        start_time = time.time()

        if self.encodec.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encodec.encoder(x)  # [B,dimension,T]
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.bandwidth)  # [n_q, B, T]
        rvq_indices = codes.detach().transpose(0, 1).contiguous()  # [B, k, T]

        encoding_time = time.time() - start_time
        # print('encoding_time:', encoding_time)
        if compress:
            context_tensor = self.get_context_tensor(context_mode, x.device, slices=n_slices)

            results = self.LM.streaming_compress_fine_layer(rvq_indices, context_tensor, self.frame_rate,
                                                  lookahead_frames, interval, max_context_frames or 50, real_compress=True)
            results['rate_b'] = self.rate_b
            results['rate_kbps_no_fec'] = self.rate_b + results['rate_f']

            enc_ec_time = time.time() - start_time
            # print('compression result', results['rate_b'], results['rate_kbps_no_fec'])

        B, K, T = rvq_indices.size()

        if vqlayer_bins is None:
            npackets_per_frame = 2
            conceal_func = self.LM.streaming_conceal
        else:
            vqlayer_bins = np.asarray(vqlayer_bins)
            npackets_per_frame = self.layers2packets(K, vqlayer_bins)
            conceal_func = self.LM.streaming_conceal_progressive

        # print("packets", vqlayer_bins, K, npackets_per_frame)

        # packet loss trace generator
        tracelen = npackets_per_frame * T
        if trace_generator:
            if isinstance(trace_generator, str):
                packet_loss_trace = np.load(trace_generator)[:tracelen]
            else:
                packet_loss_trace = np.asarray(trace_generator.generate(tracelen))
        elif sim_mode == 'ratio':
            packet_loss_trace = np.zeros(tracelen, dtype=np.int64)  # 1 for loss
            packet_loss_trace[:math.floor(tracelen * packet_loss)] = 1
            np.random.shuffle(packet_loss_trace)
        else:  # sim_mode == 'prob', i.i.d binary bernoulli(packet_loss)
            packet_loss_trace = np.random.binomial(1, packet_loss, (tracelen))

        packet_flag_pt = self.packet_flag_pt(T, npackets_per_frame, packet_loss_trace,
                                             interval, lookahead_frames,
                                             fec, fec_mode, variable=False)
        # frame-wise or two packets

        packet_flag_pt = packet_flag_pt.to(rvq_indices.device)  # bool
        assert packet_flag_pt.size(-1) == T

        x_hat_list = {}
        codes_hat_list = {}
        valid_codes_flag_list = {}
        rtf, rtflm, rtflm_entropy_only = {}, {}, {}

        start_time = time.time()
        # get received masked indices tensor
        codes_hat, valid_codes_flag = conceal_func(rvq_indices,
                                                   packet_flag_pt,
                                                   vqlayer_bins,
                                                   concealing_mode,
                                                   num_stages_to_pred,
                                                   lookahead_frames=lookahead_frames,
                                                   interval=interval,
                                                   max_context_frames=max_context_frames or 50,
                                                   compress=compress)
        decode_conceal_time = time.time() - start_time

        if valid_codes_flag is None:
            embed_hat = self.encodec.quantizer.decode(codes_hat)
        else:
            embed_hat = self.encodec.quantizer.decode_with_different_stages(codes_hat,
                                                                            valid_q_bkt_mask=valid_codes_flag.tile(
                                                                                B, 1, 1))

        x_hat = self.encodec.decoder(embed_hat)
        x_hat = x_hat[:, :, :x.size(-1)]

        if scale is not None:
            x_hat = x_hat * scale.view(-1, 1, 1)

        decode_time = time.time() - start_time

        audio_decode_time = decode_time - decode_conceal_time
        nomlm_rtf = (encoding_time + audio_decode_time) / T * self.frame_rate  # audio en/decoding only
        lm_for_entropy_coding_only_rtf = (enc_ec_time + audio_decode_time) / T * self.frame_rate if compress else 0.0
        lm_rtf = (enc_ec_time + decode_time) / T * self.frame_rate if compress else 0.0

        # print(f'RTF: {recover_base} {recover_fine} {nomlm_rtf:.3f} {lm_rtf:.3f}')
        # tag_base = '_base' if recover_base != -1 else ''
        # tag_fine = '_fine' if recover_fine else ''
        # x_hat_list['x_hat' + tag_base + tag_fine] = x_hat
        # codes_hat_list['x_hat' + tag_base + tag_fine] = codes_hat
        # valid_codes_flag_list['x_hat' + tag_base + tag_fine] = valid_codes_flag
        #
        # rtf['x_hat' + tag_base + tag_fine] = nomlm_rtf
        # rtflm['x_hat' + tag_base + tag_fine] = lm_rtf
        # rtflm_entropy_only['x_hat' + tag_base + tag_fine] = lm_for_entropy_coding_only_rtf

        x_hat_list[concealing_mode] = x_hat
        codes_hat_list[concealing_mode] = codes_hat
        valid_codes_flag_list[concealing_mode] = valid_codes_flag

        rtf[concealing_mode] = nomlm_rtf
        rtflm[concealing_mode] = lm_rtf
        rtflm_entropy_only[concealing_mode] = lm_for_entropy_coding_only_rtf

        return x_hat_list, codes_hat_list, valid_codes_flag_list, rtf, rtflm, rtflm_entropy_only

    def packet_flag_pt(self, T, npackets_per_frame, packet_loss_trace,
                       interval: int, lookahead_frames: int,
                       fec: bool = False, fec_mode: str = "01", variable=False):
        """
        packet_flag : [npackets_per_frame * T]
        all base layers zipped to one packet, others added to (npackets_per_frame - 1) packets
        :param T:
        :param packet_loss_trace: 1-d array packet_loss_trace [npackets_per_frame * T]
        :param lookahead_frames:
        :param interval:  number of frames
        :return: frame_loss_flag [2, T]
                base:[1,2,3, 4, 5, 6, 10, 11, 12
                fine: 7,8,9,13,14,15
        """
        # npackets_per_frame = packet_loss_trace.reshape(T, -1).shape[-1]

        pkt2frm = np.arange(lookahead_frames, lookahead_frames + npackets_per_frame * T, dtype=np.int32)
        # lookahead = 3
        end = npackets_per_frame * (T - lookahead_frames)
        chunk_end = end - end % (npackets_per_frame * interval)
        pkt2frm = pkt2frm[:chunk_end].reshape(-1, npackets_per_frame, interval).swapaxes(0, 1).reshape(
            npackets_per_frame, -1)

        pkt2frm = np.hstack((
            pkt2frm,
            np.vstack(
                [np.arange(((npackets_per_frame - i) * chunk_end + i * end) // npackets_per_frame + lookahead_frames,
                           ((npackets_per_frame - i - 1) * chunk_end + (
                                   i + 1) * end) // npackets_per_frame + lookahead_frames)
                 for i in range(npackets_per_frame)]),
            np.vstack(
                [np.arange(lookahead_frames),
                 np.vstack([np.arange(npackets_per_frame * T - (npackets_per_frame - i) * lookahead_frames,
                                      npackets_per_frame * T - (npackets_per_frame - i - 1) * lookahead_frames)
                            for i in range(1, npackets_per_frame)])]) if npackets_per_frame > 1 else np.arange(
                lookahead_frames).reshape(1, -1)
        ))
        pkt2frm[0] = np.roll(pkt2frm[0], lookahead_frames)
        if variable:
            return pkt2frm

        # print(pkt2frm)
        packet_flag_pt_fec = packet_loss_trace[pkt2frm]  # first row is base layer tokens  # [npackets_per_frame, T]
        # print(pkt2frm.shape, packet_flag_pt_fec.shape)
        if fec:
            lost = packet_flag_pt_fec[0]  # base layer flag
            if fec_mode == '01':
                backup_lost = packet_loss_trace[pkt2frm[1]]
                packet_flag_pt_fec[0] = np.bitwise_and(lost, backup_lost)
            elif fec_mode == '011':  # causal
                backup_lost = packet_loss_trace[pkt2frm[1]]
                backup2_lost = packet_loss_trace[pkt2frm[2]]
                packet_flag_pt_fec[0] = np.bitwise_and(np.bitwise_and(lost, backup_lost), backup2_lost)
            elif fec_mode == '0111':  # causal
                backup_lost = packet_loss_trace[pkt2frm[1]]
                backup2_lost = packet_loss_trace[pkt2frm[2]]
                backup3_lost = np.roll(lost, -1)
                packet_flag_pt_fec[0] = np.bitwise_and(np.bitwise_and(np.bitwise_and(lost, backup_lost), backup2_lost), backup3_lost)

            else:
                raise NotImplementedError('fec mode')

        packet_flag_pt_fec = torch.as_tensor(packet_flag_pt_fec, dtype=torch.bool)
        return packet_flag_pt_fec

    def parse_fec_mode(self, fec_mode: str) -> tp.List[int]:
        return list(map(lambda x: len(x), fec_mode.split('0')))  # [preceding, (current)succeeding]

    def fec_check(self, packet_loss_trace, pkt2frms, fec: bool, fec_mode) -> tp.List:
        if not fec:
            return [seg_trace[pkt2frm] for seg_trace, pkt2frm in zip(packet_loss_trace, pkt2frms)]

        packet_flag_pts = []
        for seg_trace, pkt2frm, fec_mode_ in zip(packet_loss_trace, pkt2frms, fec_mode):
            # segment
            packet_flag_pt = seg_trace[pkt2frm]  # [npackets_per_frame, T]
            base_layer_flag = packet_flag_pt[0]
            preceed_backups, succeed_backups = self.parse_fec_mode(fec_mode_)
            npc = pkt2frm.shape[0]  # npc packet per frame

            if npc - 1 < succeed_backups:
                backup_ids = np.vstack(
                    [pkt2frm[1:]] + [np.roll(pkt2frm[0], -(ofst + 1)) for ofst in range(succeed_backups - npc + 1)])
            else:
                backup_ids = pkt2frm[1:succeed_backups + 1]
            backup_effective = np.prod(seg_trace[backup_ids], axis=0)  # seg_T

            if preceed_backups > 0:
                backup_ids = np.vstack([np.roll(pkt2frm[0], (ofst + 1)) for ofst in range(preceed_backups)])
                backup_effective *= (np.prod(seg_trace[backup_ids], axis=0))

            packet_flag_pt[0] = np.bitwise_and(base_layer_flag, backup_effective)
            packet_flag_pts.append(packet_flag_pt)

        return packet_flag_pts

    def layers2packets(self, vqlayers, vqlayer_bins):
        """
        layerwise packetizing 
        :param K:
        :param bins:
        :return: number of packets per frame
        """
        cumsumlayers = np.cumsum(vqlayer_bins)
        assert vqlayers <= cumsumlayers[-1]
        return np.sum(cumsumlayers < vqlayers) + 1

    def packets2layers(self, npc, vqlayer_bins):
        return np.cumsum(vqlayer_bins)[npc]

    def variable_bandwidth_simulation(self, x, bandwidth, context_mode: str, n_slices: int, packet_loss,
                                      vqlayer_bins=None,
                                      num_stages_to_pred: int = None, fec=True, fec_mode=['01'],
                                      trace_generator=None, sim_mode='ratio', compress=False,
                                      lookahead_frames: int = 2, interval: int = 2, max_context_frames: int = None):
        """
        simulate one audio experiencing varing loss trace
        K = 24  (12 kbps)
        packet_loss = [0.05, 0.1, 0.3, 0.02]  (prob)
        bandwidth = [12, 6, 3, 12]
        fec_mode = ['01','011','0111','01']
        :return:
        """
        assert x.size(0) == 1  # B=1
        if self.encodec.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encodec.encoder(x)  # [B,dimension,T]
        codes = self.encodec.quantizer.encode(emb, self.frame_rate, self.bandwidth)  # [n_q, B, T]
        rvq_indices = codes.detach().transpose(0, 1).contiguous()  # [B, k, T]
        B, K, T = rvq_indices.size()
        assert K == 24, "12kbps required"

        vqlayer_bins = np.asarray(vqlayer_bins)
        layer2packet = {
            K: self.layers2packets(K, vqlayer_bins),
            K // 2: self.layers2packets(K // 2, vqlayer_bins),
            K // 4: self.layers2packets(K // 4, vqlayer_bins),
            K // 8: self.layers2packets(K // 8, vqlayer_bins)
        }
        packet2layer = {v: k for k, v in layer2packet.items()}

        # define loss trace
        seg_frames = [T // len(packet_loss) for _ in range(len(packet_loss))]
        seg_frames[-1] += (T - sum(seg_frames))
        # tracelen = layer2packet[K] * T

        if trace_generator is not None:
            tracelen = layer2packet[K] * T
            if isinstance(trace_generator, str):
                packet_loss_trace = np.load(trace_generator)[:tracelen]
            else:
                packet_loss_trace = np.asarray(trace_generator.generate(tracelen))
            packet_loss_trace = np.split(packet_loss_trace, np.cumsum([layer2packet[K] * seg_T for seg_T in seg_frames])[:-1])
        else:
            packet_loss_trace = [np.random.binomial(1, loss_, (layer2packet[K] * seg_T))
                                 for loss_, seg_T in zip(packet_loss, seg_frames)]

        pkt2frms = []
        for seg_loss_trace, seg_T in zip(packet_loss_trace, seg_frames):
            pkt2frms.append(self.packet_flag_pt(seg_T, layer2packet[K], seg_loss_trace,
                                                interval, lookahead_frames, variable=True))  # frame-wise or two packets

        packet_flag_pts = self.fec_check(packet_loss_trace, pkt2frms, fec, fec_mode)

        # packet_flag_pt = packet_flag_pt.to(rvq_indices.device)  # bool
        # assert packet_flag_pt.size(-1) == T
        x_hat_list = {}
        codes_hat_list = {}
        valid_codes_flag_list = {}
        seg_layers = [int(bw_ / (self.bw_per_q / 1000)) for bw_ in bandwidth]

        for recover_base, recover_fine in [(-1, False), (0, False), (0, True)]:
            codes_hat, valid_codes_flag, frame_loss_flag = self.LM.streaming_conceal_progressive_variable_bw(
                rvq_indices, seg_layers, layer2packet, seg_frames,
                packet_flag_pts, vqlayer_bins,
                recover_base, recover_fine, num_stages_to_pred,
                lookahead_frames=lookahead_frames,
                interval=interval,
                max_context_frames=max_context_frames or 50,
                compress=compress
            )
            if valid_codes_flag is None:
                embed_hat = self.encodec.quantizer.decode(codes_hat)
            else:
                embed_hat = self.encodec.quantizer.decode_with_different_stages(codes_hat,
                                                                                valid_q_bkt_mask=valid_codes_flag.tile(
                                                                                    B, 1, 1))
            x_hat = self.encodec.decoder(embed_hat)
            x_hat = x_hat[:, :, :x.size(-1)]
            if scale is not None:
                x_hat = x_hat * scale.view(-1, 1, 1)

            tag_base = '_base' if recover_base != -1 else ''
            tag_fine = '_fine' if recover_fine else ''
            x_hat_list['x_hat' + tag_base + tag_fine] = x_hat
            codes_hat_list['x_hat' + tag_base + tag_fine] = codes_hat
            valid_codes_flag_list['x_hat' + tag_base + tag_fine] = valid_codes_flag

        return x_hat_list, codes_hat_list, valid_codes_flag_list, frame_loss_flag, packet_loss_trace


def build_model(config):
    if config.lite_model: #SoundSpring-S lite model
        config.target_bandwidths = [1.5, 3., 6, 12.] 
        lm_config = dict(
            dim=config.lite_model_config['dim'],
            heads=4,
            n_q=24,  # 12kbps
            card=1024,
            num_blocks=config.lite_model_config['num_blocks'],
            linear_units=2 * config.lite_model_config['dim'],
            base_layer_n_q=3,
        )
        dim = config.lite_model_config['dim']
        n_blocks = config.lite_model_config['num_blocks']
        model_path = config.lite_model_path[n_blocks][dim] if config.task == 'test' or config.train_cfg.resume_training else None
    
    else:
        if not config.streaming:  #SoundSpring
            lm_config = dict(
                dim=768,
                heads=4,
                n_q=36,
                card=1024,
                num_blocks=24,
                linear_units=1024,
                base_layer_n_q=3,
            )
            model_path = None # TODO
            raise NotImplementedError("Specify nonstreaming model path")
        else: # streaming SoundSpring-S
            config.target_bandwidths = [1.5, 3., 6, 12.]
            lm_config = dict(
                    dim=512,
                    heads=4,
                    n_q=24,  # 12kbps
                    card=1024,
                    num_blocks=12,
                    linear_units=1024,
                    base_layer_n_q=3,
                )
            model_path = config.model_path

    model = SoundSpring_speech16k('LMConformer', config.model_config, lm_config=lm_config,
                                  streaming=config.streaming)
    
    if model_path:
        print("Loading ckpt from " + model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['generator'], strict=True)

    return model

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("../configs/test_config.yaml")

    config.lite_model = True
    model = build_model(config)