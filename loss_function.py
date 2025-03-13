import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F
import torch.nn as nn


def L1loss(x, G_x, reduction='mean'):
    return F.l1_loss(x, G_x, reduction=reduction)


def spectral_reconstruction_loss(x, G_x, config, eps=1e-6):
    # Rec in Paper
    Lf = torch.tensor([0.0], device='cuda', requires_grad=True)
    l2_loss = torch.nn.MSELoss()
    for i in config.feat_recon_window_length:
        melspec = MelSpectrogram(sample_rate=config.SAMPLE_RATE,
                                 n_fft=2 ** i,
                                 hop_length=2 ** i // 4,
                                 n_mels=config.n_mel_bin,
                                 wkwargs={"device": config.device}).to(config.device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        lossL1 = L1loss(S_x, S_G_x)
        lossL2 = l2_loss(torch.log(S_x.abs() + eps), torch.log(S_G_x.abs() + eps))
        Lf = Lf + lossL1 + lossL2

    return Lf / len(config.feat_recon_window_length)


def feat_loss(features_stft_disc_x, features_stft_disc_G_x, detach):
    """
    discriminator feature loss
    :param features_stft_disc_x:  List[feature map:tensor]
    :param features_stft_disc_G_x:
    :return:
    """
    K = len(features_stft_disc_x)  # number of discriminators
    L = len(features_stft_disc_x[0])  # number of layer
    Loss_Feat = torch.tensor([0.0], device='cuda', requires_grad=True)
    for i in range(K):
        layer_filter_x = features_stft_disc_x[i]
        layer_filter_Gx = features_stft_disc_G_x[i]
        for j in range(L):
            layer_x = layer_filter_x[j]
            layer_Gx = layer_filter_Gx[j]
            # if detach:
            #     layer_x = layer_x.detach()
            #     layer_Gx = layer_Gx.detach()
            Up_L1Loss = L1loss(layer_x, layer_Gx)
            Down_Mean = layer_x.abs().mean()
            Feat_Loss_Layer = Up_L1Loss / Down_Mean
            Loss_Feat = Loss_Feat + Feat_Loss_Layer

    return Loss_Feat / K / L


def loss_adv_g(logits_fake):
    """
    adversarial loss
    :param features_stft_disc_G_x:
    :return:
    """
    loss_g = torch.tensor([0.0], device='cuda', requires_grad=True)
    K = len(logits_fake)
    for i in range(K):
        loss_g = loss_g + F.relu(1 - logits_fake[i]).mean()
    return loss_g / K


def criterion_g(x, G_x, features_stft_disc_x, features_stft_disc_G_x, logits_fake, config, detach=False):
    """
    weighted G Loss
    :param x:
    :param G_x:
    :param features_stft_disc_x:
    :param features_stft_disc_G_x:
    :param logits_fake:
    :param config:
    :return:
    """
    if detach:
        G_x = G_x.detach()
    L1_Loss = L1loss(x, G_x)  # reduction = 'mean'
    Lf_Loss = spectral_reconstruction_loss(x, G_x, config)

    Lfeat_Loss = feat_loss(features_stft_disc_x, features_stft_disc_G_x, detach=detach)
    Lg_Loss = loss_adv_g(logits_fake)  # adversarial loss

    Loss_Total = config.LAMBDA_L1 * L1_Loss + config.LAMBDA_Lf * Lf_Loss + config.LAMBDA_Lfeat * Lfeat_Loss + config.LAMBDA_Lg * Lg_Loss

    return {'L_t': L1_Loss,
            'L_freq': Lf_Loss,
            'L_feat': Lfeat_Loss,
            'L_adv_g': Lg_Loss,
            'L_g': Loss_Total}


def criterion_adv_d(logits_real, logits_fake):
    loss_d = torch.tensor([0.0], device='cuda', requires_grad=True)
    for logits_real_, logits_fake_ in zip(logits_real, logits_fake):
        loss_d = loss_d + F.relu(1 - logits_real_).mean() + F.relu(1 + logits_fake_).mean()

    return {'loss_d': loss_d / len(logits_real)}