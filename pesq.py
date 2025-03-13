# ==========================================================================
# utility functions for PESQ (Perceptual Evaluation of Speech Quality)
# 
# REQUIREMENT: you need to have compiled PESQ and put it in the current
#              folder. we used the -Ofast optimization flag in GCC
# ==========================================================================

import re
import os
import scipy.io.wavfile as sciwav
import numpy as np
import ctypes

path_to_PESQ = None
pesq_dll = ctypes.CDLL(path_to_PESQ)
pesq_dll.pesq.restype = ctypes.c_double


def reconstruct_from_windows(windows, config):
    """ reconstruct waveform from overlapping windows """
    OVERLAP_SIZE, EXTRACT_MODE, OVERLAP_FUNC = config.OVERLAP_SIZE,\
                                               config.EXTRACT_MODE,\
                                               config.OVERLAP_FUNC
    reconstruction = []
    lastWindow = []

    for i in range(windows.shape[0]):
        r = windows[i, :]

        if (i == 0):
            reconstruction = r
        else:
            overlapLastWindow = reconstruction[-OVERLAP_SIZE:]
            overlapThisWindow = r[:OVERLAP_SIZE]
            unmodifiedPart = r[OVERLAP_SIZE:]

            overlappedPart = np.copy(overlapLastWindow)
            for j in range(OVERLAP_SIZE):
                if (EXTRACT_MODE == 1):
                    thisMult = OVERLAP_FUNC[j]
                    lastMult = OVERLAP_FUNC[j + OVERLAP_SIZE]
                else:
                    thisMult = 1.0
                    lastMult = 1.0

                # use windowing function
                overlappedPart[j] = overlapThisWindow[j] * thisMult + \
                                    overlapLastWindow[j] * lastMult

            reconstruction[-OVERLAP_SIZE:] = overlappedPart
            reconstruction = np.concatenate([reconstruction, unmodifiedPart])
    return reconstruction


def unnormalize_waveform(waveform, maxabs):
    return np.copy(waveform) * maxabs # (-1,1)  


# interface to PESQ evaluation
def run_pesq_waveforms(clean_wav, dirty_wav): # taking in two waveforms numpy 1d vector
    ### resample to 16kHz if different
    ## p862.2  return PESQ-WB MOS SCORE which uses MAPPING defined in PESQRaw2PESQWB
    clean_wav = clean_wav.astype(np.double)
    dirty_wav = dirty_wav.astype(np.double)

    return pesq_dll.pesq(ctypes.c_void_p(clean_wav.ctypes.data),
                         ctypes.c_void_p(dirty_wav.ctypes.data),
                         len(clean_wav),
                         len(dirty_wav))


# interface to PESQ evaluation, taking in two sets of windows as input
def run_pesq_windows(clean_wnd, dirty_wnd, wparam1, wparam2, config):
    clean_wnd = np.reshape(clean_wnd, (-1, config.WINDOW_SIZE))
    clean_wav = reconstruct_from_windows(clean_wnd, config)
    clean_wav = unnormalize_waveform(clean_wav, wparam1)
    clean_wav = np.clip(clean_wav, -32767, 32767)

    dirty_wnd = np.reshape(dirty_wnd, (-1, config.WINDOW_SIZE))
    dirty_wav = reconstruct_from_windows(dirty_wnd, config)
    dirty_wav = unnormalize_waveform(dirty_wav, wparam2)
    dirty_wav = np.clip(dirty_wav, -32767, 32767)

    return run_pesq_waveforms(clean_wav, dirty_wav)



def scale_MOSLQO(moslqo):
    """scales moslqo output from MOS-LQO [1.0, 4.5ish] to [0, 1]"""
    out = (moslqo - 1.0) / (4.5 - 1.0)
    return np.clip(out, 0.0, 1.0)


def PESQ2MOSLQO(pesq):
    return 0.999 + (4.999 - 0.999) / (1 + np.exp(-1.4945 * pesq + 4.6607))


def MOSLQO2PESQ(moslqo):
    """ inverse transform of PESQ2MOSLQO """
    return -(np.log((4.999 - 0.999) / (moslqo - 0.999) - 1) - 4.6607) / 1.4945

def PESQRaw2PESQWB(pesq):
    """
    defined in ITU P.862.2   same as PESQ.so
    :param pesq:
    :return:
    """
    return 0.999 + (4.999 - 0.999) / (1 + np.exp(-1.3669 * pesq + 3.8224))


# interface to PESQ evaluation, taking in two sets of windows as input
def run_pesq_windows_SoundStream(clean_wnd, dirty_wnd, maxabs, amplitude, config, **kwargs):
    """
    :param clean_wnd:
    :param dirty_wnd:
    :param maxabs:
    :param amplitude:
    :param config:
    :param kwargs:
    :return:
    """
    clean_wav = clean_wnd.flatten()
    clean_wav = clean_wav / amplitude
    clean_wav = unnormalize_waveform(clean_wav, maxabs)
    clean_wav = clean_wav / config.peak_normalized_value
    clean_wav = np.clip(clean_wav, -1.0, 1.0)

    dirty_wav = dirty_wnd.flatten()
    dirty_wav = dirty_wav / amplitude
    dirty_wav = unnormalize_waveform(dirty_wav, maxabs)
    dirty_wav = dirty_wav / config.peak_normalized_value
    dirty_wav = np.clip(dirty_wav, -1.0, 1.0)

    return run_pesq_waveforms(clean_wav, dirty_wav), clean_wav, dirty_wav


def run_pesq_windows_SoundStream_test(clean_wnd, dirty_wnd, maxabs, amplitude, config):
    clean_wnd = np.reshape(clean_wnd, (-1, config.SEG_LENGTH))
    clean_wav = reconstruct_from_windows(clean_wnd, config)
    clean_wav = clean_wav / amplitude
    clean_wav = unnormalize_waveform(clean_wav, maxabs)
    clean_wav = clean_wav / config.peak_normalized_value
    clean_wav = np.clip(clean_wav, -1.0, 1.0)

    dirty_wnd = np.reshape(dirty_wnd, (-1, config.SEG_LENGTH))
    dirty_wav = reconstruct_from_windows(dirty_wnd, config)
    dirty_wav = dirty_wav / amplitude
    dirty_wav = unnormalize_waveform(dirty_wav, maxabs)
    dirty_wav = dirty_wav / config.peak_normalized_value
    dirty_wav = np.clip(dirty_wav, -1.0, 1.0)

    return run_pesq_waveforms(clean_wav, dirty_wav), clean_wav, dirty_wav


def clip_wav(wav, limit=0.99):
    return np.clip(wav, -limit, limit)

def run_pesq_wav(wav, wav_hat):
    wav_hat = clip_wav(wav_hat)
    wav = np.copy(wav.flatten())
    wav_hat = np.copy(wav_hat.flatten())
    # print('pesq,', wav_hat.shape, wav.shape)

    return run_pesq_waveforms(wav, wav_hat), wav, wav_hat



if __name__ == '__main__':
    print(PESQ2MOSLQO(0.5))