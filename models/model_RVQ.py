import math
from pathlib import Path
import typing as tp
from modules.seanet import *
import modules as m
import quantization as qt
from utils import _linear_overlap_add, _linear_overlap_add_v0, _linear_overlap_add_v1
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from random import choice, uniform

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class Encodec_RVQ(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization. 
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """

    def __init__(self,
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float],
                 sample_rate: int,
                 channels=1,
                 normalize: bool = False,
                 segment: tp.Optional[float] = None,  # set None when inference
                 overlap: float = 0.01,  
                 name: str = 'unset'):
        super().__init__()
        self.target_bandwidths = target_bandwidths
        self.bandwidth = None
        self.encoder = encoder  # SEANetEncoder(channels=channels, n_filters=n_filters, dimension=self.out_channel_M)
        self.decoder = decoder  # SEANetDecoder(channels=channels, n_filters=n_filters, dimension=self.out_channel_M)
        self.quantizer = quantizer
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."

        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(np.array([2, 4, 5, 8])))
        self.name = name
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def overlap_length(self):
        return int(self.overlap * self.sample_rate)

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x, nframes=None):
        assert x.ndim == 3
        _, channels, length = x.size()
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor):
        duration = x.size(-1) / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        y = self.encoder(x)  # [B, dim, T]
        if self.training:
            return y, scale

        codes = self.quantizer.encode(y, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)  # codes is [B, K, T], with T frames, K： number of used quantizer.
        return codes, scale

    def decode(self, encoded_frames, nframes=None) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add_v1(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame):
        codes, scale = encoded_frame  # codes [B, K, T]
        if not self.training:
            codes = codes.transpose(0, 1)
            y_hat = self.quantizer.decode(codes)
        else:
            y_hat = codes
        out = self.decoder(y_hat)

        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x, nframes: tp.List = None, given_bandwidth=None, y_detach=False):
        frames = self.encode(x)  # list[Encodedframe]
        y_hat = []
        loss_commit = torch.tensor([0.0], device=x.device, requires_grad=True)
        bandwidth = given_bandwidth if given_bandwidth is not None else self.bandwidth
        for emb, scale in frames:
            qv = self.quantizer.forward(emb, self.frame_rate, bandwidth)
            loss_commit = loss_commit + qv.penalty
            y_hat.append((qv.quantized, scale))

        return self.decode(y_hat)[:, :, :x.size(-1)], loss_commit

    def inference(self, x, given_bandwidth=None):
        assert self.training is False
        frames = self.encode(x)  # including codes and scale
        out = self.decode(frames)
        return out[:, :, :x.size(-1)]

    def get_lm_model(self, LMclass, ckpt=None):
        """Return the associated LM model to improve the compression rate.
        """
        device = next(self.parameters()).device
        lm = LMclass(self.quantizer.n_q, self.quantizer.bins, num_layers=5, dim=200,
                     past_context=int(3.5 * self.frame_rate)).to(device)
        lm.eval()
        return lm

    def count_parameters(self) -> int:
        r"""Count parameters of encoders"""
        return sum([p.numel() for p in self.parameters()])

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   dimension: int = 128,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = False,
                   segment: tp.Optional[float] = None,
                   overlap: float = 0.01,
                   quantizer_bins: int = 1024,
                   name: str = 'unset'):
        encoder = m.SEANetEncoder(channels=channels, dimension=dimension, norm=model_norm, causal=causal,
                                  n_filters=32, ratios=[8, 5, 4, 2])
        decoder = m.SEANetDecoder(channels=channels, dimension=dimension, norm=model_norm, causal=causal,
                                  n_filters=32, ratios=[8, 5, 4, 2])
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        print('max n_q:', n_q)  
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,  
            n_q=n_q,  
            bins=quantizer_bins,  
        )
        model = Encodec_RVQ(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            overlap=overlap,
            name=name,
        )
        return model

    @staticmethod
    def _get_snake_model(target_bandwidths: tp.List[float],
                         sample_rate: int = 24_000,
                         dimension: int = 128,
                         channels: int = 1,
                         n_filters: int = 32,
                         n_residual_layers: int = 1,
                         lstm: int = 0,
                         final_activation: str = None,
                         causal: bool = True,
                         model_norm: str = 'weight_norm',
                         audio_normalize: bool = False,
                         segment: tp.Optional[float] = None,
                         overlap: float = 0.01,
                         quantizer_bins: int = 1024,
                         name: str = 'unset'):
        encoder = m.SEANetEncoder_Snake(channels=channels, dimension=dimension, norm=model_norm, causal=causal,
                                        n_residual_layers=n_residual_layers, n_filters=n_filters, ratios=[8, 5, 4, 2],
                                        lstm=lstm)
        decoder = m.SEANetDecoder_Snake(channels=channels, dimension=dimension, norm=model_norm, causal=causal,
                                        n_residual_layers=n_residual_layers, n_filters=n_filters, ratios=[8, 5, 4, 2],
                                        lstm=lstm, final_activation=final_activation)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        print('max n_q:', n_q)  
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,  
            n_q=n_q,  
            bins=quantizer_bins,  
        )
        model = Encodec_RVQ(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            overlap=overlap,
            name=name,
        )
        return model

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth

    @staticmethod
    def causal_model_18khz(config, dimension=128, pretrained: bool = False, ckpt_path: str = None,
                           lm_required=False, LMClass=None):
        """
        SEGMENT length NONE  causal True
        :param config:
        :param pretrained:
        :param ckpt_path:
        :return:
        """
        target_bandwidths = [1.5, 3., 6, 12., 18]
        sample_rate = 16000
        channels = 1

        model = Encodec_RVQ._get_model(
            target_bandwidths, sample_rate, dimension, channels,
            quantizer_bins=config.quantizer_bins,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='encodec_18khz' if pretrained else 'unset')
        model.eval()

        if pretrained:
            assert ckpt_path is not None
            print('Loading encodec from ', ckpt_path)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict['generator'])

            if lm_required:
                assert 'lm' in state_dict.keys(), "lm not included in the checkpoint"
                lm = model.get_lm_model(LMClass)
                lm.load_state_dict(state_dict['lm'])
                return model, lm
        return model

    @staticmethod
    def causal_model_6khz(config, dimension=128, pretrained: bool = False, ckpt_path: str = None):
        """
        SEGMENT length NONE  causal True
        :param config:
        :param pretrained:
        :param ckpt_path:
        :return:
        """
        target_bandwidths = [1, 2, 3, 4.5, 6]
        n_q = 12
        sample_rate = 16000
        channels = 1

        model = Encodec_RVQ._get_model(
            target_bandwidths, sample_rate, dimension, channels,
            quantizer_bins=config.quantizer_bins,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='causal_6khz' if pretrained else 'unset')
        if pretrained:
            assert ckpt_path is not None
            state_dict = torch.load(ckpt_path, map_location='cpu')['generator']
            state_dict.keys()
            unwanted = list(
                filter(lambda x: int(x.split('.')[3]) >= n_q, [x for x in state_dict.keys() if 'quantizer' in x]))
            if len(unwanted) > 0:
                for unwanted_key in unwanted:
                    del state_dict[unwanted_key]
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def noncausal_model_18kHz(config, dimension=128, pretrained: bool = False, ckpt_path: str = None):
        target_bandwidths = [1.5, 3., 6, 12., 18]
        n_q = 36
        sample_rate = 16000
        channels = 1

        model = Encodec_RVQ._get_model(
            target_bandwidths, sample_rate, dimension, channels,
            quantizer_bins=config.quantizer_bins,
            causal=False, model_norm='weight_norm', audio_normalize=False,
            name='noncausal_18khz_' if pretrained else 'unset')
        if pretrained:
            assert ckpt_path is not None
            state_dict = torch.load(ckpt_path, map_location='cpu')['generator']
            state_dict.keys()
            unwanted = list(
                filter(lambda x: int(x.split('.')[3]) >= n_q, [x for x in state_dict.keys() if 'quantizer' in x]))
            if len(unwanted) > 0:
                for unwanted_key in unwanted:
                    del state_dict[unwanted_key]
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def noncausal_model_18kHz_layernorm(config, dimension=128, pretrained: bool = False, ckpt_path: str = None):
        target_bandwidths = [1.5, 3., 6, 12., 18]
        n_q = 36
        sample_rate = 16000
        channels = 1

        model = Encodec_RVQ._get_model(
            target_bandwidths, sample_rate, dimension, channels,
            quantizer_bins=config.quantizer_bins,
            causal=False, model_norm='time_group_norm', audio_normalize=False,
            name='noncausal_18khz_' if pretrained else 'unset')
        if pretrained:
            assert ckpt_path is not None
            state_dict = torch.load(ckpt_path, map_location='cpu')['generator']
            state_dict.keys()
            unwanted = list(
                filter(lambda x: int(x.split('.')[3]) >= n_q, [x for x in state_dict.keys() if 'quantizer' in x]))
            if len(unwanted) > 0:
                for unwanted_key in unwanted:
                    del state_dict[unwanted_key]
            model.load_state_dict(state_dict)
        model.eval()
        return model



def test():
    from itertools import product
    bandwidths = [3, 6, 12, 18]
    models = {
        'encodec_18khz': Encodec_RVQ.causal_model_18khz
    }
    import librosa
    DUMMY_SIGNALS, _ = librosa.load(librosa.ex("choice"))

    for model_name, bw in product(models.keys(), bandwidths):
        model = models[model_name]()
        print(bw)
        model.set_target_bandwidth(bw)
        # wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")

        wav = torch.as_tensor(DUMMY_SIGNALS)
        # wav_hat = model.inference(wav.view(1, 1, -1))
        # print(wav.size(), wav_hat.size())

        wav_in = wav[:model.segment_length * 5].view(5, 1, model.segment_length)
        print(wav_in.size())
        wav_dec = model(wav_in, nframes=[2, 3])
        assert wav_in.shape == wav_dec.shape, (wav.shape, wav_dec.shape)
