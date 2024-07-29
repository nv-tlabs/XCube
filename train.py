# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import bdb
import importlib
import os
import pdb
import shutil
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from test import OverfitLoggerNull
from typing import List, Optional
import random

import pytorch_lightning as pl
import torch
import wandb
import yaml
from loguru import logger as loguru_logger
from omegaconf import OmegaConf
from packaging import version
from pycg import exp
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import DataParallel

from xcube.utils import wandb_util

if version.parse(pl.__version__) > version.parse('1.8.0'):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback


class CopyModelFileCallback(Callback):
    def __init__(self):
        self.source_path = None
        self.target_path = None

    def on_train_start(self, trainer, pl_module):
        if self.source_path is not None and self.target_path is not None:
            if self.target_path.parent.exists():
                shutil.move(self.source_path, self.target_path)


class CustomizedDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        inputs = self.module.module.dp_scatter(inputs, device_ids, self.dim) if inputs else []
        kwargs = self.module.module.dp_scatter(kwargs, device_ids, self.dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs


class CustomizedDataParallelPlugin(DataParallelPlugin):
    def __init__(self, parallel_devices: Optional[List[torch.device]]):
        # Parallel devices will be later populated in accelerator. Well done!
        super().__init__(parallel_devices=parallel_devices)

    def setup(self, model):
        from pytorch_lightning.overrides.data_parallel import \
            LightningParallelModule

        # model needs to be moved to the device before it is wrapped
        model.to(self.root_device)
        self._model = CustomizedDataParallel(LightningParallelModule(model), self.parallel_devices)


def determine_usable_gpus():
    if program_args.gpus is None:
        program_args.gpus = 1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        original_cvd = [int(t) for t in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        original_cvd = []

    if len(original_cvd) == program_args.gpus:
        # Everything is fine.
        return

    # Mismatched/missing CVD setting & #gpus, reset.
    gpu_states = exp.get_gpu_status("localhost")
    # temporally remove this to run multiple experiments on the same machine
    available_gpus = [t for t in gpu_states if t.gpu_mem_usage < 0.2 and t.gpu_compute_usage < 0.2]
    # available_gpus = [t for t in gpu_states]

    if len(available_gpus) == 0:
        print("You cannot use GPU. Everything is full.")
        sys.exit(0)

    if len(available_gpus) < program_args.gpus:
        print(f"Warning: Available GPUs are {[t.gpu_id for t in available_gpus]}, "
              f"but you want to use {program_args.gpus} GPUs.")
        program_args.gpus = len(available_gpus)

    available_gpus = available_gpus[:program_args.gpus]
    selection_str = ','.join([str(t.gpu_id) for t in available_gpus])
    print(f"Intelligent GPU selection: {selection_str}")
    os.environ['CUDA_VISIBLE_DEVICES'] = selection_str

def is_rank_zero():
    # It will also set LOCAL_RANK env variable, so using that will be more consistent.
    return os.environ.get('MASTER_PORT', None) is None

def is_rank_node_zero():
    return os.environ.get('NODE_RANK', '0') == '0'

def remove_option(parser, option):
    for action in parser._actions:
        if vars(action)['option_strings'][0] == option:
            parser._handle_conflict_resolve(None, [(option, action)])
            break


def readable_name_from_exec(exec_list: List[str]):
    keys = {}
    for exec_str in exec_list:
        kvs = exec_str.split("=")
        k_name = kvs[0]
        k_name_arr = ["".join([us[0] for us in t.split("_") if len(us) > 0]) for t in k_name.split(".")]
        # Collapse leading dots except for the last one.
        k_name = ''.join(k_name_arr[:-2]) + '.'.join(k_name_arr[-2:])
        k_value = kvs[1]
        if k_value.lower() in ["true", "false"]:
            k_value = str(int(k_value.lower() == "true"))
        keys[k_name] = k_value
    return '-'.join([k + keys[k] for k in sorted(list(keys.keys()))])


if __name__ == '__main__':
    """""""""""""""""""""""""""""""""""""""""""""""
    [1] Parse and initialize program arguments
        these include: --debug, --profile, --gpus, --num_nodes, --resume, ...
        they will NOT be saved for a checkpoints.
    """""""""""""""""""""""""""""""""""""""""""""""
    program_parser = exp.argparse.ArgumentParser()
    program_parser.add_argument('--debug', action='store_true', help='Use debug mode of pytorch')
    program_parser.add_argument('--resume', action='store_true', help='Continue training. Use hparams.yaml file.')
    program_parser.add_argument('--nolog', action='store_true', help='Do not create any logs.')
    program_parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    program_parser.add_argument('--save_topk', default=2, type=int, help='How many top models to save. -1 to save all models.')
    program_parser.add_argument('--validate_first', action='store_true', help='Do a full validation with logging before training starts.')
    program_parser.add_argument('--logger_type', choices=['tb', 'wandb', 'none'], default='wandb')
    program_parser.add_argument('--wname', default=None, type=str, help='Run name to be appended to wandb')
    program_parser.add_argument('--wandb_base', type=str, default="../wandb/", help="Path to wandb base directory.")
    program_parser.add_argument('--eval_interval', type=int, default=1, help='How often to evaluate the model.')
    program_parser.add_argument('--save_every', default=50, type=int, help='How often to save the model.')
    program_parser.add_argument('--resume_from_ckpt', default=None, type=str, help='checkpoint path we want to load')
    program_parser.add_argument('--model_precision', default=32, help='Model precision to use.')
    program_parser.add_argument('--seed', type=int, default=0, help='Set a random seed.')
    program_parser = pl.Trainer.add_argparse_args(program_parser)
    # Remove some args, which we think should be model-based.
    remove_option(program_parser, '--accumulate_grad_batches')
    program_args, other_args = program_parser.parse_known_args()


    model_parser = exp.ArgumentParserX(base_config_path='configs/default/param.yaml')
    model_args = model_parser.parse_args(other_args)
    hyper_path = model_args.hyper
    del model_args["hyper"]
    
    # Default logger type
    if program_args.nolog:
        program_args.logger_type = 'none'

    # AUTO resume with wandb logger
    ## uncomment if you want to use it and fill in your own <WANDB_USER_NAME>!
    # loguru_logger.info(f"This is train_auto.py! Please note that you should use 300 instead of 300.0 for resuming.")
    # if program_args.logger_type == 'wandb':
    #     wname = program_args.wname
    #     sep_pos = str(model_args.name).find('/')
    #     project_name = model_args.name[:sep_pos]
    #     run_name = model_args.name[sep_pos + 1:] + "/" + wname

    #     check_wandb_name = "<WANDB_USER_NAME>/xcube-%s/%s:last" % (project_name, run_name)
    #     try:
    #         print("Try to load from wandb:", check_wandb_name)
    #         wdb_run, args_ckpt = wandb_util.get_wandb_run(check_wandb_name, wdb_base=program_args.wandb_base, default_ckpt="last")
    #         assert args_ckpt is not None, "Please specify checkpoint version!"
    #         assert args_ckpt.exists(), "Selected checkpoint does not exist!"
    #         print("Load from wandb:", check_wandb_name)
    #         program_args.resume = True
    #         other_args[0] = check_wandb_name
    #     except:
    #         print("No wandb checkpoint found, start training from scratch")
    #         pass

    # Force not to sync to shorten bootstrap time.
    if program_args.nosync:
        os.environ['NO_SYNC'] = '1'

    # Train forever
    if program_args.max_epochs is None:
        program_args.max_epochs = -1

    if is_rank_zero():
        # Detect usable GPUs.
        determine_usable_gpus()
        # Wandb version check
        if program_args.gpus > 1 and program_args.accelerator is None:
            if version.parse(pl.__version__) > version.parse('1.8.0'):
                program_args.strategy = 'ddp'
                program_args.accelerator = "gpu"
            else:
                program_args.accelerator = 'ddp'

        if version.parse(pl.__version__) > version.parse('1.5.0'):
            program_args.devices = program_args.gpus
            del program_args.gpus
            program_args.accelerator = "gpu"
    else:
        # Align parameters.
        if version.parse(pl.__version__) > version.parse('1.8.0'):
            program_args.strategy = 'ddp'
            program_args.accelerator = "gpu"
            program_args.devices = program_args.gpus
            del program_args.gpus
        else:
            program_args.accelerator = 'ddp'


    # Profiling and debugging options
    torch.autograd.set_detect_anomaly(program_args.debug)

    # specify logdir if not use wandb
    dirpath = None
    if program_args.logger_type == 'none':
        wname = os.path.basename(hyper_path).split(".")[0]
        sep_pos = str(model_args.name).find('/')
        project_name = model_args.name[:sep_pos]
        run_name = model_args.name[sep_pos + 1:] + "_" + wname
        logdir = os.path.join("./checkpoints", project_name, run_name)
        os.makedirs(logdir, exist_ok=True)
        dirpath = logdir

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename='{epoch:06d}-{step:09d}',
        save_last=True,
        save_top_k=program_args.save_topk,
        monitor="val_step",
        mode="max",
        every_n_train_steps=program_args.save_every,
    )
    lr_record_callback = LearningRateMonitor(logging_interval='step')
    copy_model_file_callback = CopyModelFileCallback()

    # Determine parallel plugin:
    if program_args.accelerator == 'ddp':
        if version.parse(pl.__version__) < version.parse('1.8.0'):
            from pytorch_lightning.plugins import DDPPlugin
            accelerator_plugins = [DDPPlugin(find_unused_parameters=False)]
        else:
            accelerator_plugins = []
    elif program_args.accelerator == 'dp':
        accelerator_plugins = [CustomizedDataParallelPlugin(None)]
    else:
        accelerator_plugins = []

    """""""""""""""""""""""""""""""""""""""""""""""
    [2] Determine model arguments
        MODEL args include: --lr, --num_layers, etc. (everything defined in YAML)
        These use AP-X module, which accepts CLI and YAML inputs.
        These args will be saved as hyper-params.
    """""""""""""""""""""""""""""""""""""""""""""""
    if program_args.resume:
        raw_hyper = other_args[0]
        if raw_hyper.startswith("wdb:"):
            # Load config and replace
            wdb_run, wdb_ckpt = wandb_util.get_wandb_run(raw_hyper, program_args.wandb_base, default_ckpt="last")
            tmp_yaml_name = '/tmp/' + str(uuid.uuid4()) + '.yaml'
            with open(tmp_yaml_name, 'w') as outfile:
                yaml.dump(wandb_util.recover_from_wandb_config(wdb_run.config), outfile)
            other_args[0] = tmp_yaml_name

    """""""""""""""""""""""""""""""""""""""""""""""
    [3] Build / restore logger and checkpoints.
    """""""""""""""""""""""""""""""""""""""""""""""
    # Set checkpoint auto-save options.
    last_ckpt_path = None
    if program_args.resume_from_ckpt is not None:
        last_ckpt_path = program_args.resume_from_ckpt
    
    if program_args.logger_type == 'tb':
        if not program_args.resume:
            logger_version_num = None
            last_ckpt_path = None
        else:
            last_ckpt_path = Path(hyper_path).parent / "checkpoints" / "last.ckpt"
            logger_version_num = Path(hyper_path).parent.name if program_args.resume else None
        logger = TensorBoardLogger('../checkpoints/', name=model_args.name,
                                   version=logger_version_num, default_hp_metric=False)
        # Call this property to assign the version early, so we don't have to wait for the model to be loaded
        print(f"Tensorboard logger, version number =", logger.version)
    elif program_args.logger_type == 'wandb':
        # Will create wandb folder automatically
        from datetime import datetime, timedelta
        import randomname

        if not program_args.resume:
            wname = program_args.wname
            if 'WANDB_SWEEP_ID' in os.environ.keys():
                # (Use exec to determine good names)
                wname = os.environ['WANDB_SWEEP_ID'] + "-" + readable_name_from_exec(model_args.exec)
            if wname is None:
                # Example: 0105-clever-monkey
                wname = (datetime.utcnow() + timedelta(hours=8)).strftime('%m%d') + "-" + randomname.get_name()
            sep_pos = str(model_args.name).find('/')
            if sep_pos == -1:
                project_name = model_args.name
                run_name = "root/" + wname
            else:
                project_name = model_args.name[:sep_pos]
                run_name = model_args.name[sep_pos + 1:] + "/" + wname

            if is_rank_node_zero():
                logger = WandbLogger(name=run_name, save_dir=program_args.wandb_base, project='xcube-' + project_name)
            else:
                pass
        else:
            if is_rank_node_zero():
                logger = WandbLogger(name=wdb_run.name, save_dir=program_args.wandb_base, project=wdb_run.project, id=wdb_run.id)
            else:
                pass
            last_ckpt_path = wdb_ckpt
            os.unlink(tmp_yaml_name)
    else:
        logger = None


    """""""""""""""""""""""""""""""""""""""""""""""
    [4] Build trainer and determine final hparams. Set AMP if needed.
    """""""""""""""""""""""""""""""""""""""""""""""
    # Do it here because wandb name need some randomness.
    if program_args.seed == -1:
        # random seed
        program_args.seed = random.randint(0, 1000000)

    pl.seed_everything(program_args.seed)

    # Build trainer
    trainer = pl.Trainer.from_argparse_args(
        program_args,
        callbacks=[checkpoint_callback, lr_record_callback, copy_model_file_callback]
        if logger is not None else [checkpoint_callback],
        logger=logger,
        log_every_n_steps=20,
        check_val_every_n_epoch=program_args.eval_interval,
        plugins=accelerator_plugins,
        accumulate_grad_batches=model_args.accumulate_grad_batches,
        precision=program_args.model_precision)
    # fix wandb global_step resume bug
    if program_args.resume:
        # get global step offset
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
        global_step_offset = checkpoint["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
        del checkpoint    

    net_module = importlib.import_module("xcube.models." + model_args.model).Model
    net_model = net_module(model_args)

    @rank_zero_only
    def print_model_summary():
        print(" >>>> ======= MODEL HYPER-PARAMETERS ======= <<<< ")
        print(OmegaConf.to_yaml(net_model.hparams, resolve=True))
        print(" >>>> ====================================== <<<< ")
    
    print_model_summary()

    # No longer use this..
    del is_rank_zero

    """""""""""""""""""""""""""""""""""""""""""""""
    [5] Main training iteration.
    """""""""""""""""""""""""""""""""""""""""""""""
    # Note: In debug mode, trainer.fit will automatically end if NaN occurs in backward.
    e = None
    try:
        net_model.overfit_logger = OverfitLoggerNull()
        if program_args.validate_first:
            trainer.validate(net_model, ckpt_path=last_ckpt_path)
        with exp.pt_profile_named("training", "1.json"):
            trainer.fit(net_model, ckpt_path=last_ckpt_path)
    except Exception as ex:
        e = ex
        # https://stackoverflow.com/questions/52081929/pdb-go-to-a-frame-in-exception-within-exception
        if isinstance(e, MisconfigurationException):
            if e.__context__ is not None:
                traceback.print_exc()
                if program_args.accelerator is None:
                    pdb.post_mortem(e.__context__.__traceback__)
        elif isinstance(e, bdb.BdbQuit):
            print("Post mortem is skipped because the exception is from Pdb.")
        else:
            traceback.print_exc()
            if program_args.accelerator is None:
                pdb.post_mortem(e.__traceback__)

    """""""""""""""""""""""""""""""""""""""""""""""
    [6] If ended premature, add to delete list.
    """""""""""""""""""""""""""""""""""""""""""""""
    if trainer.current_epoch < 1 and last_ckpt_path is None and trainer.local_rank == 0:
        if program_args.logger_type == 'tb':
            if Path(trainer.log_dir).exists():
                with open(".premature_checkpoints", "a") as f:
                    f.write(f"{trainer.log_dir}\n")
                print(f"\n\nTB Checkpoint at {trainer.log_dir} marked to be cleared.\n\n")
            sys.exit(-1)
        elif program_args.logger_type == 'wandb':
            with open(".premature_checkpoints", "a") as f:
                f.write(f"wdb:{trainer.logger.experiment.path}:{trainer.logger.experiment.name}\n")
            print(f"\n\nWandb Run of {trainer.logger.experiment.path} "
                  f"(with name {trainer.logger.experiment.name}) marked to be cleared.\n\n")
            sys.exit(-1)

    if trainer.local_rank == 0:
        print(f"Training Finished. Best path = {checkpoint_callback.best_model_path}")