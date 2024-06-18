"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import bdb
import shutil
import sys
import re
from collections import OrderedDict, defaultdict
import numpy as np
import argparse

import yaml
import json
import pdb
import time
import pickle
import traceback
from pathlib import Path
from typing import Tuple, Union
from omegaconf import OmegaConf
import os
from contextlib import ContextDecorator
import weakref
import functools
import inspect
import logging
from rich.logging import RichHandler


# if 'MEM_PROFILE' in os.environ.keys():
#     from pytorch_memlab.line_profiler.profile import global_line_profiler
#
#     global_line_profiler.disable()

# Load and cache PT_PROFILE key
enable_pt_profile = 'PT_PROFILE' in os.environ.keys()
if enable_pt_profile and 'CUDA_LAUNCH_BLOCKING' not in os.environ.keys():
    print(" -- Warning: PT_PROFILE set but CUDA_LAUNCH_BLOCKING is not set! --")


class ConsoleColor:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def parse_config_json(json_path: Path, args: argparse.Namespace = None):
    """
    Parse a json file and add key:value to args namespace.
    Json file format [ {attr}, {attr}, ... ]
        {attr} = { "_": COMMENT, VAR_NAME: VAR_VALUE }
    """
    if args is None:
        args = argparse.Namespace()

    with json_path.open() as f:
        json_text = f.read()

    try:
        raw_configs = json.loads(json_text)
    except:
        # Do some fixing of the json text
        json_text = json_text.replace("\'", "\"")
        json_text = json_text.replace("None", "null")
        json_text = json_text.replace("False", "false")
        json_text = json_text.replace("True", "true")
        raw_configs = json.loads(json_text)

    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]
    configs = {}
    for raw_config in raw_configs:
        for rkey, rvalue in raw_config.items():
            if rkey != "_":
                configs[rkey] = rvalue

    if configs is not None:
        for ckey, cvalue in configs.items():
            args.__dict__[ckey] = cvalue
    return args


def parse_config_yaml(yaml_path: Path, args: Union[argparse.Namespace, OmegaConf] = None,
                      override: bool = True, additional_includes: list = None) -> OmegaConf:
    if args is None:
        args = OmegaConf.create()
    if isinstance(args, argparse.Namespace):
        args = OmegaConf.create(args.__dict__)

    configs = OmegaConf.load(yaml_path)
    has_include = "include_configs" in configs

    if has_include or additional_includes is not None:
        base_config_paths = []

        if has_include:
            ic_configs = configs["include_configs"]
            del configs["include_configs"]
            if isinstance(ic_configs, str):
                ic_configs = [ic_configs]
            base_config_paths += [yaml_path.parent / Path(t) for t in ic_configs]

        if additional_includes is not None:
            base_config_paths += [Path(t) for t in additional_includes]

        # Update the config from top to down.
        base_cfg = OmegaConf.create()
        for base_config_path in base_config_paths:
            base_cfg = parse_config_yaml(base_config_path, base_cfg)
        configs = OmegaConf.merge(base_cfg, configs)

    if "assign" in configs:
        overlays = configs["assign"]
        del configs["assign"]
        assign_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in overlays.items()])
        configs = OmegaConf.merge(configs, assign_config)

    if override:
        return OmegaConf.merge(args, configs)
    else:
        return OmegaConf.merge(configs, args)


def dict_to_args(data, recursive: bool = False):
    args = argparse.Namespace()
    if hasattr(data, '__dict__'):
        # This enables us to also process namespace.
        data = data.__dict__
    for ckey, cvalue in data.items():
        if recursive:
            if isinstance(cvalue, dict):
                cvalue = dict_to_args(cvalue, recursive)
        args.__dict__[ckey] = cvalue
    return args


ArgumentParser = argparse.ArgumentParser


class ArgumentParserX(ArgumentParser):
    def __init__(self, base_config_path=None, add_hyper_arg=True, to_oconf=True, **kwargs):
        super().__init__(**kwargs)
        self.add_hyper_arg = add_hyper_arg
        self.base_config_path = base_config_path
        self.to_oconf = to_oconf
        if self.add_hyper_arg:
            self.add_argument('hyper', type=str, help='Path to the yaml parameter')
        self.add_argument('--exec', type=str, nargs='+', help='Extract code to modify the args')
        self.add_argument('--include', type=str, nargs='+', help='Additional configs to include, will be appended to the include_configs key')

    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentError('Boolean value expected.')

    def parse_args(self, args=None, namespace=None, additional_args=None):
        # Parse arg for the first time to extract args defined in program.
        _args = self.parse_known_args(args, namespace)[0]
        # Add the types needed.
        file_args = OmegaConf.create()
        if self.base_config_path is not None:
            file_args = parse_config_yaml(Path(self.base_config_path), file_args)
        if additional_args is not None:
            file_args = OmegaConf.merge(file_args, additional_args)
        if self.add_hyper_arg and _args.hyper != "none":
            if _args.hyper.endswith("json"):
                file_args = parse_config_json(Path(_args.hyper), file_args)
            else:
                file_args = parse_config_yaml(Path(_args.hyper), file_args, additional_includes=_args.include)
        for ckey, cvalue in file_args.items():
            try:
                if isinstance(cvalue, bool):
                    self.add_argument(*(["--" + ckey] if ckey != "visualize" else ['-v', '--' + ckey]),
                                      type=ArgumentParserX.str2bool, nargs='?',
                                      const=True, default=cvalue)
                else:
                    self.add_argument('--' + ckey, type=type(cvalue), default=cvalue, required=False)
            except argparse.ArgumentError:
                continue
        # Parse args fully to extract all useful information
        _args = super().parse_args(args, namespace)
        if self.to_oconf:
            _args = OmegaConf.create(_args.__dict__)
        # After that, execute exec part.
        exec_code = _args.exec
        if exec_code is not None:
            for exec_cmd in exec_code:
                exec_cmd = "_args." + exec_cmd.strip()
                try:
                    exec(exec_cmd)
                except Exception as e:
                    lhs, rhs = exec_cmd.split('=')
                    exec_cmd = lhs + "='" + rhs + "'"
                    logger.warning(f"CMD boosted: {exec_cmd}")
                    exec(exec_cmd)
        return _args


class TorchLossMeter:
    """
    Weighted loss calculator, for tracing all the losses generated and print them.
    """
    def __init__(self):
        self.loss_dict = {}

    def add_loss(self, name, loss, weight=1.0):
        if weight == 0.0:
            return
        if hasattr(loss, "numel"):
            assert loss.numel() == 1, f"Loss must contains only one item, instead of {loss.numel()}."
        assert name not in self.loss_dict.items(), f"{name} already in loss!"
        self.loss_dict[name] = (weight, loss)

    def get_sum(self):
        import torch
        for n, (w, l) in self.loss_dict.items():
            if isinstance(l, torch.Tensor) and torch.isnan(l):
                print(f"Warning: Loss {n} with weight {w} has NaN loss!")
            # Disabled because this can also be used during validation/testing.
            # if l.grad_fn is None:
            #     print(f"Warning: Loss {n} with value {l} does not have grad_fn!")
        sum_arr = [w * l for (w, l) in self.loss_dict.values()]
        return sum(sum_arr)

    def items(self):
        # Standard iterator
        for n, (w, l) in self.loss_dict.items():
            yield n, w * l

    def __repr__(self):
        text = "TorchLossMeter:\n"
        for n, (w, l) in self.loss_dict.items():
            text += "   + %s: \t %.2f * %.4f = \t %.4f\n" % (n, w, l, w * l)
        text += "sum = %.4f" % self.get_sum()
        return text

    def __getitem__(self, item):
        w, l = self.loss_dict[item]
        return w * l


class AverageMeter:
    """
    Maintain named lists of numbers. Compute their average to evaluate dataset statistics.
    This can not only used for loss, but also for progressive training logging, supporting import/export data.
    """
    def __init__(self):
        self.loss_dict = OrderedDict()

    def export(self, f):
        if isinstance(f, str):
            f = open(f, 'wb')
        pickle.dump(self.loss_dict, f)

    def load(self, f):
        if isinstance(f, str):
            f = open(f, 'rb')
        self.loss_dict = pickle.load(f)
        return self

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            # loss_val = float(loss_val)
            # if np.isnan(loss_val):
            #     continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val]})
            else:
                self.loss_dict[loss_name].append(loss_val)

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            loss_dict[loss_name] = sum(loss_arr) / len(loss_arr)
        return loss_dict

    def get_mean_loss(self):
        mean_loss_dict = self.get_mean_loss_dict()
        if len(mean_loss_dict) == 0:
            return 0.0
        else:
            return sum(mean_loss_dict.values()) / len(mean_loss_dict)

    def get_printable_mean(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, loss_mean in self.get_mean_loss_dict().items():
            all_loss_sum += loss_mean
            text += "(%s:%.4f) " % (loss_name, loss_mean)
        text += " sum = %.4f" % all_loss_sum
        return text

    def get_newest_loss_dict(self, return_count=False):
        loss_dict = {}
        loss_count_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            if len(loss_arr) > 0:
                loss_dict[loss_name] = loss_arr[-1]
                loss_count_dict[loss_name] = len(loss_arr)
        if return_count:
            return loss_dict, loss_count_dict
        else:
            return loss_dict

    def get_printable_newest(self):
        nloss_val, nloss_count = self.get_newest_loss_dict(return_count=True)
        return ", ".join([f"{loss_name}[{nloss_count[loss_name] - 1}]: {nloss_val[loss_name]}"
                          for loss_name in nloss_val.keys()])

    def print_format_loss(self, color=None):
        if hasattr(sys.stdout, "terminal"):
            color_device = sys.stdout.terminal
        else:
            color_device = sys.stdout
        if color == "y":
            color_device.write(ConsoleColor.YELLOW)
        elif color == "g":
            color_device.write(ConsoleColor.GREEN)
        elif color == "b":
            color_device.write(ConsoleColor.BLUE)
        print(self.get_printable_mean(), flush=True)
        if color is not None:
            color_device.write(ConsoleColor.RESET)


class RunningAverageMeter:
    """
    new_mean = alpha * old_mean + (1 - alpha) * cur_value
        - the smaller alpha is, the more prone to change according to newest value.
    # TODO: Just merge this as a function get_running_loss_dict() / get_printable_running().
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: loss_val})
            else:
                old_mean = self.loss_dict[loss_name]
                self.loss_dict[loss_name] = self.alpha * old_mean + (1 - self.alpha) * loss_val

    def get_loss_dict(self):
        return {k: v for k, v in self.loss_dict.items()}


class AutoPdb:
    """
    Print debug info if exceptions are triggered w/o handling.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return
        if isinstance(exc_val, bdb.BdbQuit):
            logger.info("Post mortem is skipped because the exception is from Pdb.")
        else:
            # traceback.print_exc()
            logger.exception("Exception caught by AutoPdb:")
            pdb.post_mortem(exc_tb)
        # if isinstance(exc_val, MisconfigurationException):
        #     if exc_val.__context__ is not None:
        #         traceback.print_exc()
        #         if program_args.accelerator is None:
        #             pdb.post_mortem(exc_val.__context__.__traceback__)
        # elif isinstance(exc_val, bdb.BdbQuit):
        #     print("Post mortem is skipped because the exception is from Pdb.")
        # else:
        #     traceback.print_exc()
        #     if program_args.accelerator is None:
        #         pdb.post_mortem(exc_val.__traceback__)


class Timer:
    
    def __init__(self, enabled: bool = True, cuda_sync: bool = False, color: str = ConsoleColor.YELLOW):
        self.enabled = enabled
        self.cuda_sync = cuda_sync
        self.time_names = ["Timer Created"]
        self.time_points = [time.perf_counter()]
        self.report_up_to_date = False
        self.console_color = color
        self.forbid_report_on_exit = False

    def toc(self, name):
        if not self.enabled:
            return

        if self.cuda_sync:
            import torch
            torch.cuda.synchronize()

        self.time_points.append(time.perf_counter())
        self.time_names.append(name)
        self.report_up_to_date = False

    def report(self, merged=False):
        if self.enabled:
            sys.stdout.write(self.console_color)
            print("==========  TIMER  ==========")
            if not merged:
                for time_i in range(len(self.time_names) - 1):
                    print(f"{self.time_names[time_i]} --> {self.time_names[time_i + 1]}: "
                          f"{self.time_points[time_i + 1] - self.time_points[time_i]}s")
            else:
                merged_times = defaultdict(list)
                for time_i in range(len(self.time_names) - 1):
                    interval_name = f"{self.time_names[time_i]} --> {self.time_names[time_i + 1]}"
                    interval_time = self.time_points[time_i + 1] - self.time_points[time_i]
                    merged_times[interval_name].append(interval_time)
                for tname, tarr in merged_times.items():
                    print(f"{tname}: {np.mean(tarr)} +/- {np.std(tarr)}s")
            print("=============================")
            sys.stdout.write(ConsoleColor.RESET)
        self.report_up_to_date = True
        if merged:
            self.forbid_report_on_exit = True

    def toc_and_report(self, name, merged=False):
        self.toc(name)
        self.report(merged=merged)

    def __del__(self):
        if not self.report_up_to_date and not self.forbid_report_on_exit:
            self.toc_and_report("Timer Deleted")


class TimerCollections:
    def __init__(self):
        self.timers = {}
        self.is_persistent = {}

    def enable(self, name, cuda_sync: bool, persistent: bool = False):
        """
        Enable a timer, with global visibility.
        :param name: name of the timer
        :param cuda_sync: whether to sync cuda
        :param persistent: whether to really delete the timer when finalize.
        :return:
        """
        if name in self.timers.keys():
            assert persistent
        else:
            self.timers[name] = Timer(cuda_sync=cuda_sync)
            self.timers[name].forbid_report_on_exit = True
            self.is_persistent[name] = persistent
        self.timers[name].toc("activated")

    def finalize(self, name, merged=False):
        if name in self.timers.keys():
            self.timers[name].toc_and_report("Finalize", merged=merged)
            if not self.is_persistent[name]:
                del self.timers[name]
                del self.is_persistent[name]

    def toc(self, name, message):
        if name in self.timers.keys():
            self.timers[name].toc(message)

    def report(self, name):
        self.timers[name].report(merged=True)

global_timers = TimerCollections()


def natural_time(elapsed):
    if elapsed > 1.0:
        return f"{elapsed:.3f}s"
    else:
        return f"{elapsed * 1000:.3f}ms"


def performance_counter(n_iter: int = 1, run_ahead: int = 0, desc: str = None):
    """
    Usage:
    for _ in performance_counter(5):
        some_code()
    :param n_iter: number of iterations
    :param run_ahead: iter to run before timing (used to warm-up)
    """
    if desc is not None:
        print(f" + {desc} ...")
    for i_iter in range(run_ahead):
        yield i_iter - run_ahead
    all_times = []
    for i_iter in range(n_iter):
        try:
            start_time = time.perf_counter()
            yield i_iter
            all_times.append(time.perf_counter() - start_time)
        except Exception as e:
            print(e)
    print(f" + Perf Success {len(all_times)} / {n_iter}, "
          f"time = {natural_time(np.mean(all_times))} +/- {natural_time(np.std(all_times))}")


def profile(func):
    """
    Usage: (for line profiling)
        - Add a `@pycg.exp.profile` to the function you want to test
        - Run: `kernprof -l main.py`
        - Check Result: `python -m line_profiler 1.lprof`
    """
    import builtins
    # Detect kernprof line-profiler environment
    if "profile" in builtins.__dict__:
        print("Kernprof function added!")
        return builtins.__dict__["profile"](func)
    else:
        return func


class pt_profile_named(ContextDecorator):
    """
    Pytorch Profiler utility usage:
     - Annotate function with `@pycg.exp.pt_profile` or label a block using 'with pycg.exp.pt_profile_named('NAME')'
     - Run with ENV variable 'PT_PROFILE' set. (if it is 1, then only cpu profile, 2 is cuda profile)
     -  (Usually 1 will be enough even if you're on GPU because 1 will show all host cuda calls)
    """

    profiler = None

    def __init__(self, name: str, trace_file: str = "pt_profile.json"):
        self.name = name
        self.trace_file = trace_file
        self.tagger = None
        self.is_top = False

    def __enter__(self):
        if not enable_pt_profile:
            return

        from torch.profiler import profile, record_function, ProfilerActivity

        if pt_profile_named.profiler is None:
            self.is_top = True
            if int(os.environ['PT_PROFILE']) > 1:
                act = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
            else:
                act = [ProfilerActivity.CPU]
            pt_profile_named.profiler = profile(activities=act, record_shapes=True)
            pt_profile_named.profiler.__enter__()

        self.tagger = record_function(self.name)
        self.tagger.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not enable_pt_profile:
            return

        self.tagger.__exit__(exc_type, exc_val, exc_tb)
        if self.is_top:
            pt_profile_named.profiler.__exit__(exc_type, exc_val, exc_tb)
            pt_profile_named.profiler.export_chrome_trace(self.trace_file)
            pt_profile_named.profiler = None

        del self.tagger


def pt_profile(func):
    assert not isinstance(func, str), "Please use pt_profile_named."
    if enable_pt_profile:
        def new_func(*args, **kwargs):
            with pt_profile_named(func.__name__):
                return func(*args, **kwargs)
        return new_func
    else:
        return func


def readable_size(num_bytes):
    from math import isnan
    from calmsize import size as calmsize
    return '' if isnan(num_bytes) else '{:.2f}'.format(calmsize(num_bytes))


# Monkey Patch pytorch_memlab:
if 'MEM_PROFILE' in os.environ.keys():
    mem_profile_spec = os.environ['MEM_PROFILE']
    mem_threshold = 0
    if ',' in mem_profile_spec:
        mem_profile_spec, mem_threshold = mem_profile_spec.split(',')
    mem_profile_spec = int(mem_profile_spec)
    mem_threshold = int(mem_threshold)

    if mem_profile_spec == 1:

        from pytorch_memlab.line_profiler.line_records import RecordsDisplay

        def new_repr(self):
            recorded_func_names = self._line_records.index.levels[0]
            func_info = []
            for func_name in recorded_func_names:
                func_mem_info = self._line_records.loc[func_name].active_bytes
                start_mem = func_mem_info.iloc[0].item()
                peak_mem = func_mem_info.max().item()
                end_mem = func_mem_info.iloc[-1].item()
                if peak_mem - start_mem < mem_threshold * 1024 * 1024:
                    continue
                func_info.append(f"{func_name} +{readable_size(end_mem - start_mem)} "
                                 f"PEAK = +{readable_size(peak_mem - start_mem)} ({readable_size(peak_mem)})")
            return '++ (PF) ++ ' + '\n'.join(func_info) + '\n'

        RecordsDisplay.__repr__ = new_repr


def mem_profile(func=None, every=-1, active_only=True):
    """
    Usage:
        - Add a `@pycg.exp.mem_profile' to the function you want to test
        -   Optionally add parameters '@pycg.exp.mem_profile(every=1)'
        - Run the script with environment variable 'MEM_PROFILE=1,128' or 'MEM_PROFILE=2' set.
        - When the program ends, it will print the profiling result.
    """
    if 'MEM_PROFILE' in os.environ.keys():
        from pytorch_memlab import profile_every, profile
        from pytorch_memlab.line_profiler.line_profiler import DEFAULT_COLUMNS
        import functools

        if active_only:
            col = [DEFAULT_COLUMNS[0]]
        else:
            col = DEFAULT_COLUMNS
        if func is None:
            if every == -1:
                return functools.partial(profile, columns=col)
            else:
                return profile_every(output_interval=every, columns=col)
        else:
            return profile(func, columns=col)
    else:
        if func is None:
            return lambda f: f
        return func


def mem_profile_class():
    """
    Class decorator, this allows you to find out which function causes the most memory increase!
    """
    # https://stackoverflow.com/questions/16626789/functools-partial-on-class-method
    # def new_func_closure(method_name, method):
    #     import torch
    #
    #     def call_new_func(*args, **kwargs):
    #         begin_mem = torch.cuda.memory_allocated()
    #         res = method(*args, **kwargs)
    #         end_mem = torch.cuda.memory_allocated()
    #         cost_mem = (end_mem - begin_mem) / 1024 / 1024
    #         if cost_mem > mem_th_mb:
    #             print(f"Call {method_name} takes {cost_mem:.2f}MB memory. (Current = {end_mem / 1024 / 1024:.2f}MB)")
    #         return res
    #     return call_new_func

    def transform_cls(cls):
        methods = [func for func in dir(cls) if not func.startswith("_")]
        # methods = inspect.getmembers(cls, inspect.ismethod)
        for method in methods:
            old_func = getattr(cls, method)
            if not callable(old_func):
                continue
            # setattr(cls, method, new_func_closure(method, old_func))
            setattr(cls, method, mem_profile(old_func, every=1))
        return cls
    return transform_cls


def memory_usage(tensor: "torch.Tensor" = None):
    import torch

    if isinstance(tensor, torch.Tensor):
        size_mb = readable_size(tensor.element_size() * tensor.nelement())
        size_storage = readable_size(sys.getsizeof(tensor.storage()))
        return f"Torch tensor {list(tensor.size())}, logical {size_mb}, actual {size_storage}."
    elif tensor is None:
        pytorch_active = readable_size(torch.cuda.memory_allocated())
        try:
            smi_active = readable_size(get_gpu_status("localhost")[0].gpu_mem_byte)
        except Exception:
            smi_active = "-"
        return f"[Active {pytorch_active}, SMI = {smi_active}]"
    elif isinstance(tensor, str) and tensor == "lost":
        pytorch_reserved = torch.cuda.memory_reserved()
        smi_active = get_gpu_status("localhost")[0].gpu_mem_byte
        return f"[Lost {readable_size(smi_active - pytorch_reserved)}]"
    else:
        return "Memory usage not supported."


def deterministic_hash(data):
    """
    :param data: Any type
    :return: a deterministic hash value of integer type (32bit)
    """
    import zlib
    jval = json.dumps(data, ensure_ascii=False, sort_keys=True,
                      indent=None, separators=(',', ':'))
    return zlib.adler32(jval.encode('utf-8'))


# GPU monitoring stuff
class ComputeDevice:
    def __init__(self):
        self.server_name = None
        self.gpu_id = None
        self.gpu_mem_byte = None
        self.gpu_mem_total = None
        self.gpu_mem_usage = None
        self.gpu_compute_usage = None
        self.processes = []

    def __repr__(self):
        return f"{self.server_name}-GPU-{self.gpu_id}: Mem: {self.gpu_mem_usage * 100:.2f}%, " \
               f"Util: {self.gpu_compute_usage * 100:.2f}%"


class ComputeProcess:
    def __init__(self):
        self.server_name = None
        self.gpu_id = None
        self.pid = None
        self.cwd = None


def get_gpu_status(server_name, get_process_info: bool = False, use_nvml: bool = True):
    import subprocess

    def run_command(cmd):
        if cmd[0] == 'ssh' and cmd[1] == 'localhost':
            cmd = cmd[2:]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = result.stdout
        return result.decode('utf-8')

    # Use nvml if possible
    all_devs = []
    if server_name == 'localhost' and use_nvml and not get_process_info:
        import pynvml
        pynvml.nvmlInit()
        for gpu_id in range(pynvml.nvmlDeviceGetCount()):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            except pynvml.NVMLError_GpuIsLost:
                print(f"Warning: GPU {gpu_id} is lost.")
                continue
            handle_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
            handle_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cur_dev = ComputeDevice()
            cur_dev.server_name = server_name
            cur_dev.gpu_id = gpu_id
            cur_dev.gpu_compute_usage = handle_rate.gpu / 100
            cur_dev.gpu_mem_usage = handle_rate.memory / 100
            cur_dev.gpu_mem_byte = handle_memory.used
            cur_dev.gpu_mem_total = handle_memory.total
            all_devs.append(cur_dev)
        pynvml.nvmlShutdown()
        proc_info = []
    else:
        nv_output = run_command(['ssh', server_name, 'nvidia-smi'])
        nv_info = re.findall(r'(\d+)MiB / (\d+)MiB.*?(\d+)%', nv_output)
        proc_info = re.findall(r'(\d+).*?N/A.*?(\d+)\s+[CG]', nv_output)
        for gpu_id, (cur_mem, all_mem, cur_util) in enumerate(nv_info):
            cur_dev = ComputeDevice()
            cur_dev.server_name = server_name
            cur_dev.gpu_id = gpu_id
            cur_dev.gpu_mem_usage = int(cur_mem) / int(all_mem)
            cur_dev.gpu_mem_byte = cur_mem * 1024 * 1024
            cur_dev.gpu_mem_total = all_mem * 1024 * 1024
            cur_dev.gpu_compute_usage = int(cur_util) / 100
            all_devs.append(cur_dev)

    # Get current working directory...
    proc_cwds = {}
    if get_process_info:
        cwd_output = run_command(['ssh', server_name, 'pwdx', ' '.join([t[1] for t in proc_info])])
        proc_cwds_list = cwd_output.strip()
        if proc_cwds_list:
            for p in proc_cwds_list.split('\n'):
                colon_pos = p.find(': ')
                proc_cwds[int(p[:colon_pos])] = p[colon_pos + 2:]

    for proc_gpu, proc_pid in proc_info:
        proc_gpu = int(proc_gpu)
        proc_pid = int(proc_pid)
        cur_proc = ComputeProcess()
        cur_proc.server_name = server_name
        cur_proc.gpu_id = proc_gpu
        cur_proc.pid = proc_pid
        cur_proc.cwd = proc_cwds.get(proc_pid, None)
        all_devs[proc_gpu].processes.append(cur_proc)

    return all_devs


def mkdir_confirm(path: Union[Path, str]):
    if path.exists():
        assert path.is_dir()
        while True:
            ans = input(f"Directory {path} exists. [r]emove / [k]eep / [e]xit ?")
            ans = ans.lower()
            if ans == 'r':
                shutil.rmtree(path)
                break
            elif ans == 'k':
                break
            elif ans == 'e':
                raise KeyboardInterrupt

    path.mkdir(parents=True, exist_ok=True)


def lru_cache_class(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator


# class CustomFormatter(logging.Formatter):
#
#     HEADER = f"{ConsoleColor.CYAN}%(asctime)s (%(filename)s:%(lineno)d){ConsoleColor.RESET} "
#     CONTENT = "[%(levelname)s] %(message)s "
#
#     FORMATS = {
#         logging.DEBUG: HEADER + ConsoleColor.RESET + CONTENT + ConsoleColor.RESET,
#         logging.INFO: HEADER + ConsoleColor.RESET + CONTENT + ConsoleColor.RESET,
#         logging.WARNING: HEADER + ConsoleColor.YELLOW + CONTENT + ConsoleColor.RESET,
#         logging.ERROR: HEADER + ConsoleColor.RED + CONTENT + ConsoleColor.RESET,
#         logging.CRITICAL: HEADER + ConsoleColor.BOLD + ConsoleColor.RED + CONTENT + ConsoleColor.RESET
#     }
#
#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt, "%m-%d %H:%M:%S")
#         return formatter.format(record)


# Global variable for application-wise logging.
logger = logging.getLogger("pycg.exp")
logger.setLevel(logging.DEBUG)
logger.addHandler(RichHandler(markup=True, rich_tracebacks=True, log_time_format="[%m-%d %H:%M:%S]"))

# __ch = logging.StreamHandler()
# __ch.setLevel(logging.DEBUG)
# __ch.setFormatter(CustomFormatter())
# logger.addHandler(__ch)


class GlobalManager:
    """
    Manages global variables
    """
    def __init__(self):
        self.variables = {}

    def register_variable(self, name, init_value):
        self.set(name, init_value)

    def set(self, name, value):
        self.variables[name] = value

    def get(self, name):
        return self.variables[name]


global_var_manager = GlobalManager()

## pycg.image
def from_mplot(fig, close: bool = False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.pyplot as plt

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))

    if close:
        plt.close(fig)

    return image