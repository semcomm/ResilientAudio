import os
import warnings

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


from models import SoundSpring_speech16k
import scipy.signal as sig
from datetime import datetime
import torch.optim as optim
from dataset import get_loader, AudioDataset4
import torch.multiprocessing
from utils import AverageMeter, save_model, logger_configuration, write_json
import time
from loss_function import *
from pesq import *
import soundfile as sf

import random
from einops import rearrange
import wandb

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from trainer.scheduler import WarmupPolicy, NoamHoldAnnealing
from metrics import AudioMetrics


def dataset_setup(config):
    config.input_duration_sec = config.input_duration * 1e-3  
    config.SEG_LENGTH = int(config.input_duration_sec * config.SAMPLE_RATE)  # input samples length
    config.overlap_duration_sec = config.overlap_ratio * config.input_duration * 1e-3
    config.OVERLAP_SIZE = int(config.overlap_duration_sec * config.SAMPLE_RATE)
    config.STEP_SIZE = config.SEG_LENGTH - config.OVERLAP_SIZE
    config.Nframe_per_sec = config.SAMPLE_RATE / config.STEP_SIZE
    config.peak_normalized_value = 0.95

    return config

def get_accuracy(labels_bkt, logits_bckt):
    logits = rearrange(logits_bckt, 'b c k t -> b k t c')
    mask = labels_bkt.eq(-1)  # ignore_index
    maske_target = labels_bkt[~mask]
    masked_logits = logits[~mask]
    masked_token_prediction = torch.argmax(masked_logits, dim=-1)
    token_correct = (masked_token_prediction == maske_target).sum()
    token_total = maske_target.shape[0]
    return token_correct, token_total


def train_one_epoch_lm_only(model, train_loader, optimizer_lm, epoch, run, weighted):
    global global_step

    model.LM.train()
    elapsed, lm_base_loss, lm_fine_loss, lm_base_correct, lm_base_total, lm_fine_correct, lm_fine_total = [
        AverageMeter() for _ in range(7)]
    metrics = [elapsed, lm_base_loss, lm_fine_loss, lm_base_correct, lm_base_total, lm_fine_correct, lm_fine_total]

    logger.info(f" Epoch {epoch:3d}, lr {optimizer_lm.state_dict()['param_groups'][0]['lr']}")

    model.zero_grad()
    for (batch_idx, (waveform, wavlength, _, _)) in enumerate(train_loader):

        start_time = time.time()
        global_step += 1

        x = torch.as_tensor(waveform, dtype=torch.float32, device=device)

        D_index = random.randint(0, model.num_target_bandwidths - 1)

        result = model.forward(x, D_index,
                               update_D=False,
                               update_G=False,
                               lm_only=True,
                               weighted=weighted)

        result_base = result['base']
        result_fine = result['fine']

        if result_fine is not None:
            loss = result_base[0] + result_fine[0]
            lm_fine_loss.update(result_fine[0].item())

            token_correct, token_total = get_accuracy(labels_bkt=result_fine[2], logits_bckt=result_fine[1])
            lm_fine_correct.update(token_correct.item())
            lm_fine_total.update(token_total)
        else:
            loss = result_base[0]

        lm_base_loss.update(result_base[0].item())

        ### Calculate accuracy:
        token_correct, token_total = get_accuracy(labels_bkt=result_base[2], logits_bckt=result_base[1])
        lm_base_correct.update(token_correct.item())
        lm_base_total.update(token_total)

        elapsed.update(time.time() - start_time)

        # update LM
        loss.backward()
        # if global_step % train_cfg.accum_grad == 0:
        #     optimizer_lm.step()
        #     optimizer_lm.zero_grad()
        optimizer_lm.step()
        optimizer_lm.zero_grad()

        if global_step % train_cfg.print_every == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            acc_b = lm_base_correct.sum / lm_base_total.sum
            acc_f = lm_fine_correct.sum / lm_fine_total.sum
            log = (' | '.join([
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.avg:.2f}',
                f'CELoss B {lm_base_loss.avg:8.4f}',
                f'CELoss F {lm_fine_loss.avg:8.4f}',
                f'Accuracy B {acc_b:8.4f}',
                f'Accuracy F {acc_f:8.4f}'
            ]))
            logger.info(log)

            if run is not None:
                run.log({"step": global_step,
                         "accuracy_B": acc_b,
                         "accuracy_F": acc_f,
                         "CELoss B": lm_base_loss.avg,
                         "CELoss F": lm_fine_loss.avg,
                         })

            for m in metrics:
                m.clear()



def train(model, train_loader, testDataset, optimizer_g, optimizer_lm, train_mode: str = 'lm_only',
          run=None, weighted=False, **kwargs):
    global global_step

    scheduler = NoamHoldAnnealing(optimizer_lm, last_epoch=-1, **train_cfg.lr_config)

    for epoch in range(epoch_begin, train_cfg.N_EPOCHS + 1):
        if train_mode == 'lm_only':
            scheduler.step()
            train_one_epoch_lm_only(model, train_loader, optimizer_lm, epoch, run, weighted=weighted)
            # test_lm(model, testDataset, bandwidth=18, slices=5, weighted=kwargs['weighted'])

            if epoch % train_cfg.ckpt_save_every == 0 or epoch == 1:
                save_model({'generator': model.state_dict(),
                            'epoch': epoch,
                            'global_step': global_step,
                            'optim_lm': optimizer_lm.state_dict(),
                            },
                           save_path=config.models + '/Model_ep{:03d}.ckpt'.format(epoch))


if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/train_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger
    filename = datetime.now().__str__()[:-7].replace(" ", "-").replace(":", "")
    filename = filename + '_eval' if config.task != 'train' else filename
    workdir = './history/{}'.format(filename)
    config.workdir = workdir
    config.filename = filename
    config.log = workdir + '/Log_{}.log'.format(filename)
    config.samples = workdir + '/samples'
    config.pic = workdir + '/figures'
    config.models = workdir + '/models'
    config.logger = None
    print_step = 200

    logger = logger_configuration(config, save_log=config.save_log)

    if config.save_log:
        import shutil
        shutil.copy('./train_SoundSpring.py', os.path.join(config.workdir, 'train_SoundSpring.py'))
        shutil.copy('./models/SoundSpring_speech.py', os.path.join(config.workdir, 'SoundSpring_speech.py'))
        shutil.copy('./entropy_model/LM.py', os.path.join(config.workdir, 'LM.py'))

    if config.lite_model:   # only applicable for soundspring-s
        config.target_bandwidths = [1.5, 3., 6, 12.]  
        config.model_config.target_bandwidths = [1.5, 3., 6, 12.]

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
        model_path = config.lite_model_path[n_blocks][dim] if config.train_cfg.resume_training else None
    else:
        if not config.streaming:
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
            warnings.warn("not implemented")
        else:
            # for streaming  SoundSpring-S
            config.target_bandwidths = [1.5, 3., 6, 12.]
            config.model_config.target_bandwidths = [1.5, 3., 6, 12.] # todo

            lm_config = dict(
                    dim=512,
                    heads=4,
                    n_q=24,  # 12kbps
                    card=1024,
                    num_blocks=12,
                    linear_units=1024,
                    base_layer_n_q=3,
                )
            model_path = config.model_path if config.train_cfg.resume_training else None


    model = SoundSpring_speech16k('LMConformer',
                                  config.model_config,
                                  lm_config=lm_config,
                                  streaming=config.streaming)

    epoch_begin = config.train_cfg.epoch_begin
    # state_dict = torch.load(model_path)['generator']
    # print(state_dict.keys())
    # print('===============')
    # print([name for name, _ in model.named_parameters()])


    if model_path:
        logger.info("Loading ckpt from " + model_path)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['generator'], strict=True)
    else:
        state_dict = None
        config.model_config.load = True

    model = model.to(device)

    logger.info(f"Model loaded. GPU ALLOCATED: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    logger.info("Loading dataloader")

    config = dataset_setup(config)
    dataset = config.dataset
    train_loader, _, testDataset = get_loader(config,
                                              eval(config.dataset_cfg[dataset]['datasetClass']))


    if config.task == 'train':
        optimizer_g, optimizer_lm, scheduler, scheduler_d = None, None, None, None
        train_cfg = config.train_cfg
        if train_cfg.train_mode == 'ft_decoder':
            params_dict = {name: param for name, param in model.encodec.decoder.named_parameters()}
            dec_params = (params_dict[name] for name in sorted(params_dict))
            optimizer_d = optim.Adam(dec_params, lr=train_cfg.lr_enc, betas=(0.5, 0.9))
        else:
            if train_cfg.train_mode == 'full':
                params_dict = {name: param for name, param in model.encodec.named_parameters()}
                G_params = (params_dict[name] for name in sorted(params_dict))
                optimizer_g = optim.Adam(G_params, lr=train_cfg.lr_enc,
                                         betas=(0.5, 0.9))
                # scheduler = None
                scheduler = NoamHoldAnnealing(optimizer_g, last_epoch=epoch_begin - 1 if config.model_config.load else -1,
                                              **train_cfg.lr_config)
                scheduler_d = [NoamHoldAnnealing(optim_d, last_epoch=epoch_begin - 1 if config.model_config.load else -1,
                                                 **train_cfg.lr_config) for optim_d in model.multi_optimizer_d]
            # lm optimizer
            params_lm_dict = {name: param for name, param in model.LM.named_parameters() if param.requires_grad}
            lm_params = (params_lm_dict[name] for name in sorted(params_lm_dict))
            optimizer_lm = optim.Adam(lm_params, lr=5e-5, betas=(0.9, 0.96))

            if state_dict:
                optimizer_lm.load_state_dict(state_dict['optim_lm'])

        # logger.info(config.__dict__)
        if config.save_log and config.wandb:
            wandb_cfg = config.wandb_cfg
            wandb_init_kwargs = {
                'project': wandb_cfg.project_name,  
                'name': f'{wandb_cfg.run_name} {config.filename}',
                'notes': wandb_cfg.note,  # A longer description of the run
                'config': train_cfg.update(config.model_config),
                'save_code': True,
                'job_type': 'train'
            }
            run = wandb.init(**wandb_init_kwargs)
            run.log_code(".")
        else:
            run = None

        global_step = 0
        if train_cfg.train_mode == 'ft_decoder':
            train(model, train_loader, testDataset, optimizer_g, optimizer_lm,
                  optim_decoder=optimizer_d,
                  train_mode=train_cfg.train_mode,
                  scheduler=scheduler, scheduler_d=scheduler_d, run=run, weighted=False)
        else:
            train(model, train_loader, testDataset, optimizer_g, optimizer_lm,
                  train_mode=train_cfg.train_mode,
                  scheduler=scheduler, scheduler_d=scheduler_d, run=run, weighted=train_cfg.weighted_loss)

