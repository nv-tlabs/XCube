import sys
from collections import OrderedDict
import pickle
import torch

from xcube.utils.exp import ConsoleColor

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