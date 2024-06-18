# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import bdb
import os

import omegaconf

from xcube.utils import wandb_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import types
import importlib
import argparse
from pycg import exp
import pytorch_lightning as pl
from pathlib import Path

from loguru import logger as loguru_logger

class OverfitLoggerNull:
    def __init__(self):
        self.working = False

    def log_overfit_visuals(self, *args, **kwargs):
        pass


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser

if __name__ == '__main__':
    pl.seed_everything(0)

    parser = exp.ArgumentParserX(base_config_path='configs/default/param.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt', type=str, required=False, help='Path to ckpt file.')
    parser.add_argument('--weight', type=str, required=False, default='default',
                        help="Overwrite the weight defined by --ckpt. "
                             "Explicitly set to 'none' so that no weight will be loaded.")
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--record', nargs='*', help='Whether or not to store evaluation data. add name to specify save path.')
    parser.add_argument('--focus', type=str, default="none", help='Sample to focus')
    parser.add_argument('--wandb_base', type=str, default="../wandb/", help="Path to wandb base directory.")

    known_args = parser.parse_known_args()[0]
    args_ckpt = known_args.ckpt
    if args_ckpt.startswith("wdb:"):
        wdb_run, args_ckpt = wandb_util.get_wandb_run(args_ckpt, wdb_base=known_args.wandb_base, default_ckpt="test_auto", is_train=False)
        assert args_ckpt is not None, "Please specify checkpoint version!"
        assert args_ckpt.exists(), "Selected checkpoint does not exist!"
        model_args = omegaconf.OmegaConf.create(wandb_util.recover_from_wandb_config(wdb_run.config))
    elif args_ckpt is not None:
        model_yaml_path = Path(known_args.ckpt).parent.parent / "hparams.yaml"
        model_args = exp.parse_config_yaml(model_yaml_path)
    else:
        model_args = None
    args = parser.parse_args(additional_args=model_args)

    if args.nosync:
        # Force not to sync to shorten bootstrap time.
        os.environ['NO_SYNC'] = '1'

    if args.gpus is None:
        args.gpus = 1

    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**args), logger=None, max_epochs=1)
    net_module = importlib.import_module("xcube.models." + args.model).Model

    # --ckpt & --weight logic:
    if args.weight == 'default':
        ckpt_path = args_ckpt
    elif args.weight == 'none':
        ckpt_path = None
    else:
        ckpt_path = args.weight
        
    try:
        if ckpt_path is not None:
            net_model = net_module.load_from_checkpoint(ckpt_path, hparams=args, strict=False)
        else:
            net_model = net_module(args)
        net_model.overfit_logger = OverfitLoggerNull()

        with exp.pt_profile_named("trainer.test", "1.json"):
            test_result = trainer.test(net_model)

        # Usually, PL will output aggregated test metric from LoggerConnector (obtained from trainer.results)
        #   However, as we patch self.log for test. We would print that ourselves.
        net_model.print_test_logs()

    except Exception as ex:
        if isinstance(ex, bdb.BdbQuit):
            print("Post mortem is skipped because the exception is from Pdb. Bye!")
        elif isinstance(ex, KeyboardInterrupt):
            print("Keyboard Interruption. Program end normally.")
        else:
            import sys, pdb, traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
            sys.exit(-1)
