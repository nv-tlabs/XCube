import glob
import os.path
import re
from datetime import datetime
from pathlib import Path

import flatten_dict
import torch
import wandb
from loguru import logger
from xcube.utils.exp import ConsoleColor

def _rebuild_cfg(d):
    if isinstance(d, dict):
        return {k: _rebuild_cfg(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_rebuild_cfg(t) for t in d]
    elif isinstance(d, str):
        if d == 'None':
            return None
        elif len(d) > 0 and d[0] == '{':
            # Deeper keys are collapsed. We need to rescue them:
            return eval(d)
        return d
    else:
        return d

def recover_from_wandb_config(wandb_dict):
    f_dict = flatten_dict.unflatten(wandb_dict, splitter='path')
    f_dict = _rebuild_cfg(f_dict)
    return f_dict

def get_epoch_checkpoints(ckpt_base: Path, match_case: str = r"epoch=(\d+)"):
    all_files = list(ckpt_base.glob("*.ckpt"))
    all_ckpts = {}
    for f in all_files:
        match_res = re.match(match_case, f.name)
        if match_res is not None:
            all_ckpts[int(match_res[1])] = f
    return all_ckpts

def find_latest_ckpt_file(ckpt_path):
    # Get a list of all files with .ckpt extension
    files = glob.glob(os.path.join(ckpt_path, "*.ckpt"))

    # Check if there are no .ckpt files
    if not files:
        return None

    # Find the latest file based on modification time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_wandb_run(wdb_url: str, wdb_base: str, default_ckpt: str = "all", is_train=True):
    """
    wdb_url: str. Can be two formats (See README.md for details):
        - [wdb:]<USER-NAME>/<PROJ-NAME>/<RUN-ID>[:ckpt-id]
        - [wdb:]<USER-NAME>/<PROJ-NAME>/<RUN-NAME>[:ckpt-id]
    Note that for the latter case, multiple runs can be returned and user is prompted to select one.
    If ckpt-id is provided, possible path to the checkpoints will also be provided.
    """
    if wdb_url.startswith("wdb:"):
        wdb_url = wdb_url[4:]

    if ":" in wdb_url:
        wdb_url, ckpt_name = wdb_url.split(":")
    else:
        ckpt_name = default_ckpt

    wdb_url = wdb_url.split("/")
    user_name = wdb_url[0]
    project_name = wdb_url[1]
    other_name = "/".join(wdb_url[2:])

    # Heuristic to guess user input type.
    if len(other_name) == 8 and "/" not in other_name and "_" not in other_name and "-" not in other_name:
        run_id = f"{user_name}/{project_name}/{other_name}"
        wdb_run = wandb.Api().run(run_id)
    else:
        all_runs = wandb.Api().runs(
            f"{user_name}/{project_name}",
            filters={"display_name": other_name}
        )
        assert len(all_runs) > 0, f"No run is found for name {other_name}"
        sel_idx = 0
        if len(all_runs) > 1:
            print("\nMore than 1 runs found")
            for rid, run in enumerate(all_runs):
                print(f"   {rid}: {run}")
            sel_idx = int(input("Please select one:"))
        wdb_run = all_runs[sel_idx]
        print("Target run", wdb_run)

    if ckpt_name is None:
        return wdb_run, None

    ckpt_base = Path(f"{wdb_base}/{wdb_run.project}/{wdb_run.id}/checkpoints/")

    if ckpt_name == "last":
        ckpt_path = ckpt_base / "last.ckpt"
        assert ckpt_path.exists(), f"{ckpt_path} not exist!"

        # try to check if it is complete
        if is_train:
            try:
                _ = torch.load(ckpt_path, map_location="cpu")
            except:
                # incomplete ckpt, try to find the lastest one
                # remove last.ckpt
                files = glob.glob(os.path.join(ckpt_base, "*.ckpt"))
                if str(ckpt_path) in files:
                    files.remove(str(ckpt_path))
                # Find the latest file based on modification time
                latest_file = max(files, key=os.path.getmtime)
                ckpt_path = Path(latest_file)
                print(f'{ConsoleColor.YELLOW} Warning: The existing "last" checkpoint is incomplete, '
                    f'but actually the newest one should be {latest_file}! {ConsoleColor.RESET}')
        
        existing_mtime = os.path.getmtime(ckpt_path)
        try:
            last_mtime = wdb_run.history(keys=['val_loss', '_timestamp'], samples=1000).iloc[-1]._timestamp
            if last_mtime - existing_mtime > 600:       # 10min threshold
                print(f'{ConsoleColor.YELLOW} Warning: The existing "last" checkpoint has timestamp '
                    f'{datetime.fromtimestamp(existing_mtime).isoformat()}, '
                    f'but actually the newest one should have timestamp '
                    f'{datetime.fromtimestamp(last_mtime).isoformat()}! {ConsoleColor.RESET}')
        except:
            logger.warning(f"Cannot get last timestamp from {wdb_run}")
            
    elif ckpt_name == "last_auto":
        # automatically select the lastest ckpt from ckpt_base
        ckpt_path = find_latest_ckpt_file(ckpt_base)
        assert ckpt_path is not None, f"{ckpt_base} does not contain any ckpt file!"
        # make it to path
        ckpt_path = Path(ckpt_path)
        print("Using ckpt_path", ckpt_path)
    
    elif ckpt_name == "best" or ckpt_name == "epoch":
        epoch_ckpts = get_epoch_checkpoints(ckpt_base)
        loss_table = wdb_run.history(keys=['val_loss', 'epoch'], samples=1000)
        loss_table.set_index('epoch', inplace=True)
        best_epoch_idx = loss_table['val_loss'].astype(float).argmin().item()
        if best_epoch_idx in epoch_ckpts:
            ckpt_path = epoch_ckpts[best_epoch_idx]
        else:
            print(f'{ConsoleColor.YELLOW} Best epoch is {best_epoch_idx} but not found, existed: {list(epoch_ckpts.keys())} ')
            print(loss_table)
            print('')
            raise FileNotFoundError

    elif ckpt_name == "all":
        ckpt_path = ckpt_base

    elif ckpt_name == "test_auto":
        # Try to load last, if best and last do not align, then prompt
        epoch_ckpts = get_epoch_checkpoints(ckpt_base)
        if len(epoch_ckpts) == 0:
            print(f'{ConsoleColor.YELLOW} Please check the folder of {ckpt_base}, {list(ckpt_base.glob("*.ckpt"))} {ConsoleColor.RESET}')
            raise FileNotFoundError
        existing_last_epoch_idx = max(list(epoch_ckpts.keys()))
        loss_table = wdb_run.history(keys=['val_loss', 'epoch'], samples=1000)
        loss_table.set_index('epoch', inplace=True)
        best_epoch_idx = loss_table['val_loss'].argmin().item()
        last_epoch_idx = loss_table.iloc[-1].name.item()
        if existing_last_epoch_idx == best_epoch_idx == last_epoch_idx:
            print(f'{ConsoleColor.YELLOW} Warning: ckpt not specified, but {last_epoch_idx} (being both last and best) '
                  f'seems to be a valid choice. {ConsoleColor.RESET}')
            ckpt_path = epoch_ckpts[existing_last_epoch_idx]
        else:
            print(f'{ConsoleColor.YELLOW} Cannot determine correct ckpt_idx! Best = {best_epoch_idx}, Last = {last_epoch_idx}, '
                  f'Existing Last = {existing_last_epoch_idx} {ConsoleColor.RESET}')
            assert False

    elif ckpt_name.isdigit():
        epoch_idx = int(ckpt_name)
        epoch_ckpts = get_epoch_checkpoints(ckpt_base)
        if epoch_idx in epoch_ckpts:
            ckpt_path = epoch_ckpts[epoch_idx]
        else:
            print(f'{ConsoleColor.YELLOW} Epoch {epoch_idx} not found, existed: {list(epoch_ckpts.keys())} {ConsoleColor.RESET}')
            raise FileNotFoundError

    else:
        raise NotImplementedError

    return wdb_run, ckpt_path

def find_mismatched_keys(
    state_dict,
    model_state_dict,
    loaded_keys,
    ignore_mismatched_sizes,
):
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for checkpoint_key in loaded_keys:
            model_key = checkpoint_key

            if (
                model_key in model_state_dict
                and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
            ):
                mismatched_keys.append(
                    (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                )
                del state_dict[checkpoint_key]
    return mismatched_keys

def load_state_dict_into_model(model_to_load, state_dict):
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix=""):
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)

    return error_msgs