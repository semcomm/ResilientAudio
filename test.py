import os
import numpy as np

from models import SoundSpring_speech16k
import scipy.signal as sig
from datetime import datetime
from dataset import get_loader, AudioDataset4
import torch.multiprocessing
from utils import AverageMeter, save_model, logger_configuration, write_json
import time
from pesq import *
import soundfile as sf

import random
from einops import rearrange
from metrics import AudioMetrics

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import argparse
from omegaconf import OmegaConf


def dataset_setup(config):
    config.input_duration_sec = config.input_duration * 1e-3  # SOUNDSTREAM 24K 360ms   16k 360ms  5760
    config.SEG_LENGTH = int(config.input_duration_sec * config.SAMPLE_RATE)  # input samples length
    config.overlap_duration_sec = config.overlap_ratio * config.input_duration * 1e-3
    config.OVERLAP_SIZE = int(config.overlap_duration_sec * config.SAMPLE_RATE)
    config.STEP_SIZE = config.SEG_LENGTH - config.OVERLAP_SIZE
    config.Nframe_per_sec = config.SAMPLE_RATE / config.STEP_SIZE
    config.peak_normalized_value = 0.95

    return config

def test_lm_compress(model, testDataset, bandwidth, lookahead_frames, interval, max_context_frames,
                     context_mode, slices=5, streaming=False):
    model.eval()
    model.set_target_bandwidth(bandwidth)
    elapsed, accuracy_b, accuracy_f, rate_kbps, rate_b_kbps, rate_f_kbps, rtf_enc, rtf_enc_ec = [AverageMeter() for _ in
                                                                                                 range(8)]
    logger.info(f'--------------- Test LM -- bandwidth {bandwidth} ----- slices {slices} -----------------')

    if streaming:
        inference_func = lambda x: model.streaming_inference(x,
                                                             context_mode=context_mode,
                                                             slices=slices,
                                                             lookahead_frames=lookahead_frames,
                                                             interval=interval,
                                                             max_context_frames=max_context_frames)
    else:
        inference_func = lambda x: model.inference(x, context_mode=context_mode, slices=slices)

    for i in range(len(testDataset)):
        waveform, length, _, _ = testDataset[i]
        x = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,T]
        start_time = time.time()

        out = inference_func(x)
        elapsed.update(time.time() - start_time)
        rate_kbps.update(out['rate_kbps_no_fec'])
        rate_b_kbps.update(out['rate_b'])
        rate_f_kbps.update(out['rate_f'])

        # for streaming inference
        rtf_enc.update(out['rtf_enc'])
        rtf_enc_ec.update(out['rtf_enc_ec'])

        # accuracy_b.update(out['accuracy_b'])
        if out['accuracy_f'] is not None:
            accuracy_f.update(out['accuracy_f'])
        if i % 20 == 0:
            logger.info(' | '.join([
                f'accuracy_b {accuracy_b.avg: .3f}',
                f'accuracy_f {accuracy_f.avg: .3f}',
                f'rate_kbps {rate_kbps.avg: .3f}',
                f'rate_b {rate_b_kbps.avg: .3f}',
                f'rate_f {rate_f_kbps.avg: .3f}',
                f'rtf_enc {rtf_enc.avg: .3f}',
                f'rtf_enc_ec {rtf_enc_ec.avg: .3f}',
            ]))

    logger.info(("Finished"))
    logger.info(' | '.join([
        f'runtime {elapsed.avg: .3f}',
        f'accuracy_b {accuracy_b.avg: .3f}',
        f'accuracy_f {accuracy_f.avg: .3f}',
        f'rate_b {rate_b_kbps.avg: .3f}',
        f'rate_f {rate_f_kbps.avg: .3f}',
        f'rate_kbps_w.o.fec {rate_kbps.avg: .3f}',
        f'rtf_enc {rtf_enc.avg: .3f}',
        f'rtf_enc_ec {rtf_enc_ec.avg: .3f}',
    ]))

def simulate_pl(model, testDataset, packet_loss: float = None, slices: int = 5,
                context_mode: str = 'temporal_stride', vqlayer_bins=None,
                save=False, save_interval=300, epoch=None,
                bandwidth: float = None,
                sim_mode: str= 'ratio',
                concealing_mode:str = 'base',
                num_stages_to_pred: int = None,
                fec: int = None, fec_mode: str = '101',
                stream_cfg=None, compress=False,trace_generator=None):
    logger.info(f'-------- slices: {slices:2d}  packet_loss {packet_loss}  '
                f'concealing_mode {concealing_mode} stages_pred {num_stages_to_pred} fec_baselayer {fec}  fec_mode {fec_mode}  -----------------')
    assert concealing_mode in ['base', 'all', 'none']
    model.eval()
    elapsed = AverageMeter()
    eval_pred_right, eval_pred_total,test_pesq, test_sisnr, test_sdr, test_mfcc, test_plcmos = [AverageMeter() for _ in range(7)]
    wandb_audio_list = []
    if bandwidth is None:
        bandwidth = model.encodec.target_bandwidths[-1]
    model.encodec.set_target_bandwidth(bandwidth)
    print(model.bandwidth)
    logger.info(f'-------- Simulate packet loss -- bandwidth {bandwidth}  -----')

    print('BANDWIDTH: ', model.bandwidth)
    if packet_loss is None:
        packet_loss = random.random()  # (0,1)
    metrics = ["SISNR", "SDR", "MFCC", "PLCMOS", "PESQ"]
    metric_funcs = AudioMetrics(metrics=metrics, config=config, device=device)

    if save:
        sample_dir = config.samples + f"/{bandwidth}_{packet_loss}_{concealing_mode}"
        os.mkdir(sample_dir)

    if stream_cfg:
        sim_func =  model.simulate_streaming_example
    else:
        raise NotImplementedError
    
    for i in range(len(testDataset) // 3):  # 2620 TODO
        waveform, length, _, _ = testDataset[i]
        x = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,T]
        x_hat_list, *_ = \
            sim_func(x,
                    context_mode=context_mode,
                    n_slices=slices,
                    packet_loss=packet_loss,
                    sim_mode=sim_mode,
                    vqlayer_bins=vqlayer_bins,
                    num_stages_to_pred=num_stages_to_pred,
                    fec=fec,
                    fec_mode=fec_mode,
                    compress=False,
                    concealing_mode=concealing_mode,
                    trace_generator=None,
                    **stream_cfg)
        
        scores, dirty_wav = metric_funcs(metrics, x_hat=x_hat_list[concealing_mode], x=x)
        
        test_pesq.update(scores['PESQ'])
        test_sisnr.update(scores['SISNR'])
        test_sdr.update(scores['SDR'])
        test_mfcc.update(scores['MFCC'])
        test_plcmos.update(scores['PLCMOS'])
        
        if i % 20 == 0:
            process = (i % len(testDataset)) / (len(testDataset)) * 100.0
            if packet_loss > 0:
                logger.info(' | '.join([
                    f'Step [{i % len(testDataset)}/{len(testDataset)}={process:.2f}%]',
                    f'bandwidth {bandwidth:.2f}',
                    # f'acc {eval_pred_right.sum/eval_pred_total.sum: .3f}',
                    f'Avg_PESQ {test_pesq.avg:.2f}',
                    f'Avg_SISNR {test_sisnr.avg:.2f}',
                    f'Avg_SDR {test_sdr.avg:.2f}',
                    f'Avg_mfcc_dist {test_mfcc.avg:.2f}',
                    f'Avg_plcmos {test_plcmos.avg:.2f}',
                ]))
            else:
                logger.info(' | '.join([
                    f'bandwidth {bandwidth:.2f}',
                    f'Avg_PESQ {test_pesq.avg:.2f}',
                ]))

        if save and i % save_interval == 0:
            if epoch is not None: # for evaluation in training 
                PATH = config.samples + "/Ep{}_Num{:03d}_band{:.1f}.wav".format(epoch,
                                                                                     i,
                                                                                     bandwidth)
            else:
                PATH = sample_dir + "/N{:03d}_pesq{:.2f}_sisnr{:.2f}_sdr{:.2f}_plcmos{:.2f}.wav".format(
                    i, scores['PESQ'], scores['SISNR'], scores['SDR'], scores['PLCMOS'])

                sf.write(PATH, dirty_wav, config.SAMPLE_RATE, 'PCM_16')

                PATH = sample_dir + '/Num{:03d}_ref.wav'.format(i)
                x_raw = x.detach().cpu().numpy()

                try:
                    sf.write(PATH, np.copy(x_raw.flatten()), config.SAMPLE_RATE, 'PCM_16')
                    # List = [None, 'Ref', i, wandb.Audio(PATH, sample_rate=config.SAMPLE_RATE)]
                    # wandb_audio_list.append(List)
                except:
                    logger.error('Origin audio save failed.')

            # # =======  wandb log ==========
            # List = [epoch, 'RVQ', i, wandb.Audio(PATH, sample_rate=config.SAMPLE_RATE)]
            # wandb_audio_list.append(List)

    logger.info(' | '.join([
        f'Test finished \n',
        f'bandwidth {bandwidth:.2f}',
        f'Avg_PESQ {test_pesq.avg:.2f}',
        f'Avg_SISNR {test_sisnr.avg:.2f}',
        f'Avg_SDR {test_sdr.avg:.2f}',
        f'Avg_mfcc_dist {test_mfcc.avg:.2f}',
        f'Avg_plcmos {test_plcmos.avg:.2f}',
        f'runtime {elapsed.avg: .3f}',
    ]))

    result = {
        "test_pesq": test_pesq.avg,
        "test_sisnr": test_sisnr.avg,
        "test_sdr": test_sdr.avg,
        "test_plcmos": test_plcmos.avg,
    }

    return result, wandb_audio_list

def simulate_oneaudio(model, testDataset, packet_loss: float = None, sim_mode: str = 'ratio',
                      slices: int = 5,
                      context_mode: str = 'temporal_stride', vqlayer_bins=None,
                      save=True, bandwidth: float = None, num_stages_to_pred: int = None, fec: int = None,
                      fec_mode: str = '101', compress=False, trace_generator=None, stream_cfg=None, variable_cfg=None):
    logger.info(f'-------- Simulate packet loss -----')
    logger.info(
        f'-------- slices: {slices:2d}  packet_loss {packet_loss}  stages_pred {num_stages_to_pred}  fec_baselayer {fec}  fec_mode {fec_mode} -----------------')
    model.eval()
    if bandwidth is None:
        bandwidth = model.encodec.target_bandwidths[-1]
    model.encodec.set_target_bandwidth(bandwidth)
    print('BANDWIDTH: ', model.bandwidth)
    if packet_loss is None:
        packet_loss = random.random()  # (0,1)
    metrics = ["SISNR", "SDR", "MFCC", "PLCMOS", "PESQ"]
    metric_funcs = AudioMetrics(metrics=metrics, config=config, device=device)

    if save:
        os.mkdir(config.samples + f"/{bandwidth}_{packet_loss}")

    if stream_cfg is None:
        sim_func = model.simulate_packet_loss_example
    else:
        sim_func = model.simulate_streaming_example if variable_cfg is None else model.variable_bandwidth_simulation

    # type_list = ['x_hat', 'x_hat_base', 'x_hat_base_fine']
    concealing_mode = ['base']
    pesq_summary, mfcc_summary, sisnr_summary, sdr_summary, plcmos_summary = {type: [] for type in concealing_mode}, \
                                                                             {type: [] for type in concealing_mode}, \
                                                                             {type: [] for type in concealing_mode}, {
                                                                                 type: [] for type in concealing_mode}, {
                                                                                 type: [] for type in concealing_mode}

    rtf_lm_summary = {type: [] for type in concealing_mode}
    rtf_lm_entropy_only_summary = {type: [] for type in concealing_mode}
    rtf_nonlm_summary = []

    # i = 189
    # i=9
    i = 97
    waveform, length, _, _ = testDataset[i]
    x = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,T]

    if variable_cfg and variable_cfg['trace_generator'] is not None:
        simnum = 1
    else:
        simnum = 500

    for j in range(simnum):
        if variable_cfg is None:
            x_hat_list, codes_list, valid_codes_flag_list, rtf, rtflm, rtflm_entropy_only = sim_func(x,
                                                                                 context_mode=context_mode,
                                                                                 n_slices=slices,
                                                                                 packet_loss=packet_loss,
                                                                                 sim_mode=sim_mode,
                                                                                 vqlayer_bins=vqlayer_bins,
                                                                                 num_stages_to_pred=num_stages_to_pred,
                                                                                 fec=fec,
                                                                                 fec_mode=fec_mode,
                                                                                 compress=compress,
                                                                                 trace_generator=trace_generator,
                                                                                 **stream_cfg)

        else:
            x_hat_list, _, valid_codes_flag_list, frame_loss_flag, packet_loss_trace = model.variable_bandwidth_simulation(
                x, bandwidth=variable_cfg['bandwidth'],
                context_mode=context_mode,
                n_slices=slices,
                packet_loss=variable_cfg['packet_loss'],
                sim_mode="ratio",
                vqlayer_bins=vqlayer_bins,
                num_stages_to_pred=num_stages_to_pred,
                fec=variable_cfg['fec'],
                fec_mode=variable_cfg['fec_mode'],
                compress=compress,
                trace_generator=variable_cfg['trace_generator'],
                **stream_cfg)
            np.save(config.samples + '/N{:03d}_{:04d}_frame_flag.npy'.format(i, j),
                    frame_loss_flag.detach().cpu().numpy())
            np.save(config.samples + '/N{:03d}_{:04d}_packet_loss_trace.npy'.format(i, j),
                    np.concatenate(packet_loss_trace))

        x_raw = x.detach().cpu().numpy()
        if j == 0 and save:
            PATH = config.samples + '/ref_Num{:03d}.wav'.format(i)
            sf.write(PATH, np.copy(x_raw.flatten()), config.SAMPLE_RATE, 'PCM_16')

        for type in concealing_mode:
            rtf_lm_summary[type].append(rtflm[type])
            rtf_lm_entropy_only_summary[type].append(rtflm_entropy_only[type])

            x_hat = x_hat_list[type]
            scores, dirty_wav = metric_funcs(metrics, x_hat=x_hat, x=x)

            if save and j < 100:
                PATH = config.samples + "/{}_{}/N{:03d}_{:04d}_{}_pesq{:.2f}_sisnr{:.2f}_sdr{:.2f}_plcmos{:.2f}.wav".format(
                    bandwidth, packet_loss, i, j, type[6:],
                    scores['PESQ'], scores['SISNR'], scores['SDR'], scores['PLCMOS'])
                sf.write(PATH, dirty_wav, config.SAMPLE_RATE, 'PCM_16')

            pesq_summary[type].append(scores['PESQ'])
            mfcc_summary[type].append(scores['MFCC'])
            sisnr_summary[type].append(scores['SISNR'])
            sdr_summary[type].append(scores['SDR'])
            plcmos_summary[type].append(scores['PLCMOS'])

        if j % 1 == 0:
            for type in concealing_mode:
                logger.info(f"Sample {j+1}: {type}, rtf: {np.mean(rtf_lm_summary[type]):.2f}, rtf_entropy_only: {np.mean(rtf_lm_entropy_only_summary[type]):.2f} "
                            f"pesq: {np.mean(pesq_summary[type]):.2f}, plcmos: {np.mean(plcmos_summary[type]):.2f}")


    np.save(config.samples + '/pesq.npy', pesq_summary)
    np.save(config.samples + '/mfcc.npy', mfcc_summary)
    np.save(config.samples + '/sisnr.npy', sisnr_summary)
    np.save(config.samples + '/sdr.npy', sdr_summary)
    np.save(config.samples + '/plcmos.npy', plcmos_summary)

                                    slices: int = 5,
                                    context_mode: str = 'temporal_stride', vqlayer_bins=None,
                                    save=True, bandwidth: float = None, num_stages_to_pred: int = None, fec: int = None,
                                    fec_mode: str = '101', compress=False, trace_generator=None, stream_cfg=None):
    logger.info(f'-------- Simulate packet loss -----')
    logger.info(
        f'-------- slices: {slices:2d}  packet_loss {trace_generator or packet_loss}  stages_pred {num_stages_to_pred}  fec_baselayer {fec}  fec_mode {fec_mode} -----------------')
    model.eval()

    if bandwidth is None:
        bandwidth = model.encodec.target_bandwidths[-1]
    model.encodec.set_target_bandwidth(bandwidth)
    print(model.bandwidth)
    if packet_loss is None:
        packet_loss = random.random()  # (0,1)

    metrics = ["SISNR", "SDR", "MFCC", "PLCMOS", "PESQ"]
    metric_funcs = AudioMetrics(metrics=metrics, config=config, device=device)

    if save:
        os.mkdir(config.samples + f"/{bandwidth}_{packet_loss}")

    type_list = ['x_hat', 'x_hat_base', 'x_hat_base_fine']  # 'x_hat_fine',

    # test_pesq, test_mfcc_dist, test_sisnr, test_sdr, test_plcmos
    avger = {type: [AverageMeter() for _ in range(5)] for type in type_list}

    if stream_cfg is None:
        sim_func = model.simulate_packet_loss_example
    else:
        sim_func = model.simulate_streaming_example

    userstudy_id = [12, 16, 41, 97, 193, 274, 321, 399, 734, 805]

    # for i in range(len(testDataset) // 3):  # 2620
    for i in userstudy_id:
        waveform, length, _, _ = testDataset[i]
        x = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,T]
        # start_time = time.time()
        # pesq_sample = {type: [] for type in type_list}

        for j in range(1):
            x_hat_list, codes_list, valid_codes_flag_list = sim_func(x,
                                                                     context_mode=context_mode,
                                                                     n_slices=slices,
                                                                     packet_loss=packet_loss,
                                                                     sim_mode=sim_mode,
                                                                     vqlayer_bins=vqlayer_bins,
                                                                     num_stages_to_pred=num_stages_to_pred,
                                                                     fec=fec,
                                                                     fec_mode=fec_mode,
                                                                     compress=compress,
                                                                     trace_generator=trace_generator,
                                                                     **stream_cfg)

            x_raw = x.detach().cpu().numpy()
            if save:
                PATH = config.samples + '/ref_Num{:03d}.wav'.format(i)
                sf.write(PATH, np.copy(x_raw.flatten()), config.SAMPLE_RATE, 'PCM_16')
            # pesq_j = []

            for type in type_list:
                x_hat = x_hat_list[type]
                scores, dirty_wav = metric_funcs(metrics, x_hat=x_hat, x=x)

                if save and i < 1000:
                    PATH = config.samples + "/{}_{}/N{:03d}_{:04d}_{}_pesq{:.2f}_sisnr{:.2f}_sdr{:.2f}_plcmos{:.2f}.wav".format(
                        bandwidth, packet_loss, i, j, type[6:],
                        scores['PESQ'], scores['SISNR'], scores['SDR'], scores['PLCMOS'])
                    sf.write(PATH, dirty_wav, config.SAMPLE_RATE, 'PCM_16')

                avger[type][0].update(scores['PESQ'])
                avger[type][1].update(scores['MFCC'])
                avger[type][2].update(scores['SISNR'])
                avger[type][3].update(scores['SDR'])
                avger[type][4].update(scores['PLCMOS'])

                # pesq_summary[type].append(pesq)
                # mfcc_summary[type].append(mfcc_dist.item())
                # sisnr_summary[type].append(sisnr_.item())
                # sdr_summary[type].append(sdr_.item())
                # plcmos_summary[type].append(plcmos_)

                # logger.info(f'pesq: {pesq_j[0]:.3f} {pesq_j[1]:.3f} {pesq_j[2]:.3f} {pesq_j[3]:.3f}')
            # [pesq_summary[type].append(np.mean(pesq_sample[type])) for type in type_list]

        if i % 20 == 0:
            process = (i % len(testDataset)) / (len(testDataset)) * 100.0
            logger.info(' | '.join([
                f'Step [{i % len(testDataset)}/{len(testDataset)}={process:.2f}%]',
                f'bandwidth {bandwidth:.2f}',
                f'Avg_PESQ {avger["x_hat"][0].avg:.2f} {avger["x_hat_base_fine"][0].avg:.2f}',
                f'Avg_mfcc_dist {avger["x_hat"][1].avg:.2f} {avger["x_hat_base_fine"][1].avg:.2f}',
                f'Avg_SISNR {avger["x_hat"][2].avg:.2f} {avger["x_hat_base_fine"][2].avg:.2f}',
                f'Avg_SDR {avger["x_hat"][3].avg:.2f} {avger["x_hat_base_fine"][3].avg:.2f}',
                f'Avg_plcmos {avger["x_hat"][4].avg:.2f} {avger["x_hat_base_fine"][4].avg:.2f}',
            ]))

        #logger.info(str([np.mean(pesq_summary[type]) for type in type_list]))
        # break

    # np.save(config.samples+'/pesq.npy', pesq_summary)
    # np.save(config.samples+'/mfcc.npy', mfcc_summary)
    # np.save(config.samples+'/sisnr.npy', sisnr_summary)
    # np.save(config.samples+'/sdr.npy', sdr_summary)
    # np.save(config.samples+'/plcmos.npy', plcmos_summary)

    logger.info(' | '.join([
        f'bandwidth {bandwidth:.2f}',
        f'Avg_PESQ {avger["x_hat"][0].avg:.2f} {avger["x_hat_base"][0].avg:.2f} {avger["x_hat_base_fine"][0].avg:.2f}',
        f'Avg_mfcc_dist {avger["x_hat"][1].avg:.2f} {avger["x_hat_base"][1].avg:.2f} {avger["x_hat_base_fine"][1].avg:.2f}',
        f'Avg_SISNR {avger["x_hat"][2].avg:.2f} {avger["x_hat_base"][2].avg:.2f} {avger["x_hat_base_fine"][2].avg:.2f}',
        f'Avg_SDR {avger["x_hat"][3].avg:.2f} {avger["x_hat_base"][3].avg:.2f} {avger["x_hat_base_fine"][3].avg:.2f}',
        f'Avg_plcmos {avger["x_hat"][4].avg:.2f} {avger["x_hat_base"][4].avg:.2f} {avger["x_hat_base_fine"][4].avg:.2f}',
    ]))
    # logger.info(str([np.mean(pesq_summary[type]) for type in type_list]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/test_config.yaml", help="config filepath")
    parser.add_argument('--test_concealment', action='store_true', help="test concealment performance only")
    parser.add_argument('--test_compression', action='store_true', help="test compression performance only")
    
    parser.add_argument('--wlan', action='store_true', help="wlan channel")
    # parser.add_argument('--wlan_trace', type=str, default=None, help="a given wlan channel trace")
    parser.add_argument('--vb', action='store_true', help="variable bandwidth")


    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # logger
    filename = datetime.now().__str__()[:-7].replace(" ", "-").replace(":", "")
    filename = filename + '_eval'
    workdir = './history/{}'.format(filename)
    config.workdir = workdir
    config.filename = filename
    config.log = workdir + '/Log_{}.log'.format(filename)
    config.samples = workdir + '/samples'
    config.pic = workdir + '/figures'
    config.models = workdir + '/models'
    config.logger = None
    logger = logger_configuration(config, save_log=config.save_log)

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
            model_path = None
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

    config.eval_cfg.vqlayer_bins = [base * lm_config["base_layer_n_q"] for base in config.eval_cfg.vqlayer_bins]
    print(f"cuda:{config.device_id}")
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")

    if model_path:
        logger.info("Loading ckpt from " + model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['generator'], strict=True)

    model.eval()
    model = model.to(device)

    logger.info("Loading dataloader")
    config = dataset_setup(config)
    dataset = config.dataset
    train_loader, _, testDataset = get_loader(config, eval(config.dataset_cfg[dataset]['datasetClass']))
    
    eval_cfg = config.eval_cfg
    for item in eval_cfg:
        logger.info(item)
        logger.info(eval_cfg[item])


    if args.test_concealment:
        simulate_pl(model, testDataset, 
                    concealing_mode='base', 
                    save_interval=100,
                    stream_cfg=config.stream_cfg,
                    **config.eval_cfg)

    if args.vb:
        simulate_oneaudio(model, testDataset, stream_cfg=config.stream_cfg, variable_cfg=config.variable_cfg, **eval_cfg)

    if args.wlan:
        if os.path.exists(config.trace_path):
            trace_generator = config.trace_path
        elif config.trace_path == 'wlan':
            from utils import Three_state_Markov_wlan_packet_tracer
            trace_generator = Three_state_Markov_wlan_packet_tracer()
        else:
            # random memoryless loss trace
            trace_generator = None

        simulate_oneaudio(model, testDataset,
                          stream_cfg=config.stream_cfg,
                          variable_cfg=None,
                          trace_generator=trace_generator,
                          **config.eval_cfg)


    if args.test_compression:
        test_lm_compress(model, testDataset,
                         bandwidth=eval_cfg['bandwidth'],
                         slices=eval_cfg['slices'],
                         context_mode=eval_cfg['context_mode'],
                         streaming=config.streaming,
                         **config.stream_cfg)

    print('Done')
