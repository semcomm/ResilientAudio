from pesq import run_pesq_wav

class AudioMetrics(object):
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
    from distortion import perceptual_distance  # mfcc
    from PLCMOS.plc_mos import PLCMOSEstimator

    registered_metrics = {
        "SISNR": ScaleInvariantSignalNoiseRatio,
        "SDR": SignalDistortionRatio,
        "MFCC": perceptual_distance,
        "PLCMOS": PLCMOSEstimator,
        "PESQ": run_pesq_wav,
    }

    def __init__(self, metrics, config, device):
        self.config = config
        self.metric = {}
        for metric in metrics:
            assert metric in self.registered_metrics
            if metric == "MFCC":
                self.metric[metric] = self.registered_metrics[metric](config).to(device)
            elif metric == "PLCMOS":
                self.metric[metric] = self.registered_metrics[metric]()
            elif metric == "PESQ":
                self.metric[metric] = self.registered_metrics[metric]
            else:
                self.metric[metric] = self.registered_metrics[metric]().to(device)

    def __call__(self, metrics, x_hat, x, *args, **kwargs):
        result = {}
        if "SISNR" in metrics:
            result["SISNR"] = self.metric["SISNR"](x_hat, x).item()
        if "SDR" in metrics:
            result["SDR"] = self.metric["SDR"](x_hat, x).item()
        if "MFCC" in metrics:
            cut = x.size(-1) % self.config.WINDOW_SIZE
            result["MFCC"] = self.metric["MFCC"](x[..., :-cut] if cut > 0 else x,
                                                 x_hat[..., :-cut] if cut > 0 else x_hat).item()
        if "PESQ" in metrics:
            x_raw = x.detach().cpu().numpy()
            x_hat = x_hat.squeeze().detach().cpu().numpy()  # must be mono
            result["PESQ"], _, dirty_wav = self.metric["PESQ"](x_raw, x_hat)
        if "PLCMOS" in metrics:
            result["PLCMOS"] = self.metric["PLCMOS"].run(x_hat, sr_degraded=16000,
                                                         audio_clean=None)  # input should be 16kHz, mono, [-1, 1] range

        return result, dirty_wav
