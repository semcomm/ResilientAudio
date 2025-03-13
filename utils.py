# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Various utilities."""

from hashlib import sha256
from pathlib import Path
import typing as tp
import os
import torch
import torchaudio
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_host_ip():
    """
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

def logger_configuration(config, save_log=False, test_mode=False, distributed_rank=0):
    name = "SoundSpring"
    if distributed_rank > 0:
        logger_not_root = logging.getLogger(name=name)
        logger_not_root.propagate = False
        return logger_not_root

    # logger
    logger = logging.getLogger(name)
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    # config.logger = logger
    # return config.logger
    return logger

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _linear_overlap_add(frames: torch.Tensor, stride: int, weight):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames.device
    dtype = frames.dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    seg_length = frames.shape[-1]

    # t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1] #size: [frame_length]
    # weight = 0.5 - (t - 0.5).abs()  # triangle

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    # print('weight:', weight)
    # print('total_size:', total_size)
    # print('stride:', stride)
    for i in range(frames.size(0)):
        out[..., offset:offset + seg_length] += weight * frames[i]
        sum_weight[offset:offset + seg_length] += weight
        offset += stride
        # print(offset)

    assert sum_weight.min() > 0, f'total_size: {total_size} frame_length: {seg_length}'
    return out / sum_weight

def _linear_overlap_add_v0(frames: tp.List[torch.Tensor], stride: int, weight):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    seg_length = frames[0].shape[-1]

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.size(-1)
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride

    assert sum_weight.min() > 0, f'total_size: {total_size} frame_length: {seg_length}'
    return out / sum_weight


def _linear_overlap_add_v1(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]
    # print(len(frames), frames[-1].shape)
    # print('total size', total_size)
    frame_length = frames[0].shape[-1]

    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1] #size: [frame_length]
    weight = 0.5 - (t - 0.5).abs()  # triangle

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for nof,frame in enumerate(frames):
        frame_length = frame.size(-1)
        # print(nof+1, offset, offset+frame_length, total_size, frame_length, frame.size())
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride

    assert sum_weight.min() > 0
    return out / sum_weight


def _get_checkpoint_url(root_url: str, checkpoint: str):
    if not root_url.endswith('/'):
        root_url += '/'
    return root_url + checkpoint


def _check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, 'rb') as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum:
        raise RuntimeError(f'Invalid checksum for file {path}, '
                           f'expected {checksum} but got {actual_checksum}')


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def save_model(model, save_path):
    torch.save(model, save_path)
    print('Model Save Success Path Is',save_path)


def unnormalize_waveform(waveform, maxabs):
    return np.copy(waveform) * maxabs  # (-1,1)  


def plot_indices(indices_hat: torch.Tensor, indices_target: torch.Tensor):
    """
    visualize
    :param indices_hat:
    :param indices_target:
    :return:
    """
    if indices_hat.ndim() == 3:
        indices_hat = indices_hat[0]
        indices_target = indices_target[0]
    assert indices_hat.ndim() == 2, 'plot 2d image only'

    accuracy = torch.sum(indices_hat == indices_target) / indices_target.numel()
    print('accuracy:',accuracy)

    plt.imshow((indices_hat == indices_target).to(torch.int32).numpy())
    plt.title(f'Accuracy {accuracy: .2f}')


import random

class GE_Channel_Model(object):
    """Gilbert-Elliott Channel Model"""

    def __init__(self, p, q):
        """

        :param p: prob from 0 to 1
        :param q: prob from 1 to 0
        """
        self.p = p
        self.q = q
        self.state = random.randint(0, 1)

    @property
    def next_state(self):
        while True:
            if self.state:
                self.state -= (random.uniform(0, 1) < self.q)
            else:
                self.state += (random.uniform(0, 1) < self.p)
            yield self.state

    def generate(self, length) -> tp.List:
        return [next(self.next_state) for _ in range(length)]


class Three_state_Markov_wlan_packet_tracer(object):
    """
    Ben Milner and Alastair James "An Analysis of Packet Loss Models for Distributed Speech Recognition"
    Q > Qprime
    enables long duration periods of no loss to be modeled in state 1 while state 3 models short periods of no loss which occur in-between packet loss
    """

    def __init__(self):
        self.alpha = 0.139
        self.beta = 1.69
        self.Q = 0.9363         # state 1 self-loop
        self.P = 1 - self.Q     # state 1 to 2(loss)
        self.Qprime = 0.5662    # state 3 self-loop

        self.p = 0.3631  # return to no loss
        self.q = 0.4072  # prob of consecutive packet loss
        self.state = random.randint(1, 3)  # state 1, 3: good state no packet loss  # state 2 : packet loss

        print("Markov Avg loss rate:", self.avg_packet_loss_rate)
        print("Markov Avg loss burst length:", self.avg_packet_loss_burst_loss_length)

    @property
    def avg_packet_loss_rate(self):
        return (1 - self.Qprime) * (1 - self.Q) / (
                    self.Q * (self.p + self.q + self.Qprime - 2) - self.Qprime * (1 + self.p) - self.q + 2)

    @property
    def avg_packet_loss_burst_loss_length(self):
        return 1. / (1 - self.q)

    @property
    def next_state(self):
        while True:
            if self.state == 1:
                self.state += (random.uniform(0, 1) < self.P)
            elif self.state == 2:
                rng = random.uniform(0, 1)
                self.state -= (rng < self.p)
                self.state += (rng > self.p + self.q)
            else:  # state == 3
                self.state -= (random.uniform(0, 1) > self.Qprime)
            yield self.state

    def generate(self, length) -> tp.List:
        return [next(self.next_state) == 2 for _ in range(length)]  # 1 for loss, 0 for good


import json


def write_json(python_dict, path):
    with open(path, 'w') as fp:
        json.dump(python_dict, fp)

from audiotools import AudioSignal
from audiotools.metrics.quality import *

def init_visqol(mode):
    from visqol.pb2 import similarity_result_pb2
    from visqol.pb2 import visqol_config_pb2
    from visqol import visqol_lib_py
    config = visqol_config_pb2.VisqolConfig()

    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    return api


def realtime_metric_evaluation(fs, x, x_hat, step_ms=500, eval_win_ms=2000,
                               metric=['visqol'], **kwargs):
    """
    same length x x_hat  streaming mode
    :param metric:   visqol need to specify mode "speech" or "audio"
    :param eval_win_ms: eval context
    :param step_ms:  eval stride
    :param x:   [1, C, T]
    :param x_hat:
    :return:
    """
    step_len = int(fs * step_ms / 1000)
    eval_win_len = int(fs * eval_win_ms / 1000)
    audio_len = x.size(-1)
    print(x.size())
    # batch_size = (audio_len - eval_win_len) // step_len
    # refs = torch.cat([x[..., i * step_len:i * step_len + eval_win_len] for i in range(batch_size)], dim=0)
    # ests = torch.cat([x_hat[..., i * step_len:i * step_len + eval_win_len] for i in range(batch_size)], dim=0)

    batch_size = (audio_len - eval_win_len) // step_len
    refs = [x[..., max(0, i * step_len):eval_win_len + i * step_len] for i in range(1, batch_size)]
    ests = [x_hat[..., max(0, i * step_len):eval_win_len + i * step_len] for i in range(1, batch_size)]

    print(refs[0][0].size())
    scores = {}
    if 'visqol' in metric:
        api = init_visqol(mode='speech')
        moslqo = []

        for i in range(batch_size - 1):
            _visqol = api.Measure(
                refs[i][0].detach().cpu().numpy().astype(float),
                ests[i][0].detach().cpu().numpy().astype(float),
            )
            moslqo.append(_visqol.moslqo)

        scores['visqol'] = np.array(moslqo)

    if 'pesq' in metric:
        assert fs == 16000
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
        from CommonModules.loss.pesq import PESQ2MOSLQO
        pesqmetric = PerceptualEvaluationSpeechQuality(fs, 'wb')  # 16000
        pesqs = []
        for i in range(batch_size - 1):
            pesqs.append(pesqmetric(ests[i], refs[i]).item())  # [-0.5, 4.5]
        scores['pesq'] = PESQ2MOSLQO(np.array(pesqs))  # [1.0, 4.5]

    if 'sdr' in metric:
        from torchmetrics.audio import SignalDistortionRatio
        SDR = SignalDistortionRatio()
        sdrs = []
        for i in range(batch_size - 1):
            sdr_ = SDR(ests[i], refs[i]).item()
            sdrs.append(sdr_)
        scores['sdr'] = np.asarray(sdrs)

    if 'sisnr' in metric:
        from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
        SISNR = ScaleInvariantSignalNoiseRatio()
        sisnrs = []
        for i in range(batch_size - 1):
            sisnr_ = SISNR(ests[i], refs[i]).item()
            sisnrs.append(sisnr_)
        scores['sisnr'] = np.asarray(sisnrs)

    if 'plcmos' in metric:
        from CommonModules.loss.PLCMOS.plc_mos import PLCMOSEstimator
        estimator = PLCMOSEstimator()
        plcmoss = []
        for i in range(batch_size - 1):
            plcmoss.append(estimator.run(ests[i].squeeze().numpy(), sr_degraded=16000, audio_clean=None))
        scores['plcmos'] = np.asarray(plcmoss)

    return scores

# def plot_loss_pattern(plt, pattern):
#
#     plt.imshow(data)
#     plt.show()