import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from numpy.fft import fft, ifft
from scipy.fftpack import dct, idct
import numpy as np

EPSILON = 1e-7
# ====================================================================
#  DCT (Discrete Cosine Transform)
# ====================================================================

# generate square dct matrix
#     how to use: generate n-by-n matrix M. then, if you have a signal w, then:
#                 dct(w) = M * w
#     where w must be n-by-1
#
#     backed by scipy
def generate_dct_mat(n, norm='ortho'):
    return torch.tensor(dct(np.eye(n), norm=norm))


# given a (symbolic Keras) array of size M x A
#     this returns an array M x A where every one of the M samples has been independently
#     filtered by the DCT matrix passed in
def dct(x, dct_mat):
    # reshape x into 2D array, and perform appropriate matrix operation
    reshaped_x = x.reshape(-1, dct_mat.shape[0])
    return torch.mm(reshaped_x, dct_mat)


# ====================================================================
#  DFT (Discrete Fourier Transform)
# ====================================================================

# generate two square DFT matrices, one for the real component, one for
# the imaginary component
#     dimensions are: n x n
def generate_dft_mats(n):
    mat = np.fft.fft(np.eye(n))
    return torch.tensor(np.real(mat)),\
           torch.tensor(np.imag(mat))


# generate two NON-square DFT matrices, one for the real component, one for
# the imaginary component, using np.fft.rfft
#     dimensions are: n x (fft_size / 2 + 1)
def generate_real_dft_mats(n, fft_size):
    mat = np.fft.rfft(np.eye(n), fft_size)
    return torch.tensor(np.real(mat), dtype=torch.float32),\
           torch.tensor(np.imag(mat), dtype=torch.float32)


# given array of size M x WINDOW_SIZE
#     this returns an array M x WINDOW_SIZE where every one of the M samples has been replaced by
#     its DFT magnitude, using the DFT matrices passed in
def dft_mag(x, real_mat, imag_mat):
    reshaped_x = x.reshape(-1, real_mat.shape[0])
    real = torch.mm(reshaped_x, real_mat)
    imag = torch.mm(reshaped_x, imag_mat)

    mag = torch.sqrt(torch.square(real) + torch.square(imag) + EPSILON)
    return mag



# ====================================================================
#  MFCC (Mel Frequency Cepstral Coefficients)
# ====================================================================

# based on a combination of this article:
#     http://practicalcryptography.com/miscellaneous/machine-learning/...
#         guide-mel-frequency-cepstral-coefficients-mfccs/
# and some of this code:
#     http://stackoverflow.com/questions/5835568/...
#         how-to-get-mfcc-from-an-fft-on-a-signal

# conversions between Mel scale and regular frequency scale
def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)


def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)


# generate Mel filter bank
def melFilterBank(numCoeffs, sample_rate, fftSize=None, window_size=512):
    minHz = 0
    maxHz = sample_rate / 2  # max Hz by Nyquist theorem
    if (fftSize is None):
        numFFTBins = window_size
    else:
        numFFTBins = fftSize / 2 + 1

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    numCoeffs = int(numCoeffs)
    numFFTBins = int(numFFTBins)

    # we need (numCoeffs + 2) points to create (numCoeffs) filterbanks
    melRange = np.arange(numCoeffs + 2)
    melRange = melRange.astype(np.float32)

    # create (numCoeffs + 2) points evenly spaced between minMel and maxMel
    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel
    melCenterFilters = melCenterFilters.astype(np.int32)

    for i in range(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = math.floor(numFFTBins * melCenterFilters[i] / maxHz)

        # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, numFFTBins))

    # generate triangular filters (in frequency domain)
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)

        startRange = melCenterFilters[i - 1]
        midRange = melCenterFilters[i]
        endRange = melCenterFilters[i + 1]

        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))

        filterMat[i - 1] = filter

    # return filterbank as matrix
    return filterMat


# ====================================================================
#  Finally: a perceptual loss function (based on Mel scale)
# ====================================================================

# given a (symbolic Theano) array of size M x WINDOW_SIZE
#     this returns an array M x N where each window has been replaced
#     by some perceptual transform (in this case, MFCC coeffs)


class perceptual_distance(torch.nn.Module):
    """
      perceptual loss function  (MFCC LOSS)
    """
    def __init__(self, config, **kwargs):
        super(perceptual_distance, self).__init__()
        self.window_size = config.WINDOW_SIZE
        self.FFT_SIZE = 512
        # multi-scale MFCC distance
        MEL_SCALES = [8, 16, 32, 128]
        
        # precompute Mel filterbank: [FFT_SIZE x NUM_MFCC_COEFFS]
        self.MEL_FILTERBANKS = []
        for scale in MEL_SCALES:
            filterbank_npy = melFilterBank(scale, config.SAMPLE_RATE, self.FFT_SIZE).transpose()
            self.MEL_FILTERBANKS.append(torch.tensor(filterbank_npy, dtype=torch.float32))

        # we precompute matrices for MFCC calculation
        DFT_REAL, DFT_IMAG = generate_real_dft_mats(config.WINDOW_SIZE, self.FFT_SIZE)
        self.register_buffer("DFT_REAL", DFT_REAL)
        self.register_buffer("DFT_IMAG", DFT_IMAG)
        self.register_buffer("eps", torch.tensor(1e-7))

    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(-1, self.window_size)
        y_pred = y_pred.reshape(-1, self.window_size)

        pvec_true = self.perceptual_transform(y_true)
        pvec_pred = self.perceptual_transform(y_pred)

        distances = []
        for i in range(0, len(pvec_true)):
            error = torch.unsqueeze(self.rmse(pvec_pred[i], pvec_true[i]), dim=-1)
            distances.append(error)
        distances = torch.cat(distances, dim=-1)

        loss = torch.mean(distances)
        return loss

    def rmse(self, y_true, y_pred):
        mse = torch.mean(torch.square(y_pred - y_true))
        return torch.sqrt(mse + self.eps)

    def perceptual_transform(self, x):
        transforms = []
        powerSpectrum = torch.square(dft_mag(x, self.DFT_REAL, self.DFT_IMAG))
        powerSpectrum = 1.0 / self.FFT_SIZE * (powerSpectrum + self.eps)

        for filterbank in self.MEL_FILTERBANKS:
            filteredSpectrum = torch.mm(powerSpectrum, filterbank.to(powerSpectrum.device))
            filteredSpectrum = torch.log(filteredSpectrum + self.eps)
            transforms.append(filteredSpectrum)
        return transforms
