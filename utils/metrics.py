from typing import Sequence
import torch
import numpy as np

class Metric(torch.nn.Module):
    def __init__(self, name, device=torch.device("cpu")):
        super().__init__()
        self.name = name
        self.device = device
        self.required_output_keys = ()

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        pass

    def get_output(self, reduce=True):
        pass

    def set_device(self, device):
        self.device = device

    def sync_across_processes(self, accelerator):
        pass

    def __call__(self, output):
        self.update(output)
        return self.compute()
    
class Accuracy(Metric):
    def __init__(self, name, device=torch.device("cpu")):
        super().__init__(name, device)
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)
        self.device = device

    def reset(self):
        self.correct = torch.tensor(0, device=self.device)
        self.total = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.correct = self.correct.to(device)
        self.total = self.total.to(device)

    def update(self, output):
        y_pred, batch = output
        y_true = batch['prompt']
        self.correct += (y_pred == y_true).sum().item()
        self.total += len(y_true)

    def sync_across_processes(self, accelerator):
        self.correct = accelerator.reduce(self.correct)
        self.total = accelerator.reduce(self.total)

    def compute(self):
        return {self.name:self.correct / self.total}

    def get_output(self, reduce=True):
        return self.compute()

class Precision(Metric):
    def __init__(self, name="precision", device=torch.device("cpu")):
        super().__init__(name, device)
        self.tp = torch.tensor(0, device=device)
        self.fp = torch.tensor(0, device=device)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fp = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] with 0 or 1 predicted classes
        batch["prompt"]: Tensor of shape [B] with ground truth 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        # True Positives: (pred=1, true=1)
        tp_mask = (y_pred == 1) & (y_true == 1)
        # False Positives: (pred=1, true=0)
        fp_mask = (y_pred == 1) & (y_true == 0)

        self.tp += tp_mask.sum()
        self.fp += fp_mask.sum()

    def sync_across_processes(self, accelerator):
        self.tp = accelerator.reduce(self.tp)
        self.fp = accelerator.reduce(self.fp)

    def compute(self):
        tp = self.tp.float()
        fp = self.fp.float()
        denom = tp + fp
        if denom == 0:
            precision = 0.0
        else:
            precision = tp / denom
        return {self.name: precision}
    
    def get_output(self, reduce=True):
        return self.compute()


class Recall(Metric):
    def __init__(self, name="recall", device=torch.device("cpu")):
        super().__init__(name, device)
        self.tp = torch.tensor(0, device=device)
        self.fn = torch.tensor(0, device=device)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fn = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.tp = self.tp.to(device)
        self.fn = self.fn.to(device)

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] with 0 or 1
        y_true: ground truth (batch["prompt"]), shape [B] with 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        # True Positives: (pred=1, true=1)
        tp_mask = (y_pred == 1) & (y_true == 1)
        # False Negatives: (pred=0, true=1)
        fn_mask = (y_pred == 0) & (y_true == 1)

        self.tp += tp_mask.sum()
        self.fn += fn_mask.sum()

    def sync_across_processes(self, accelerator):
        self.tp = accelerator.reduce(self.tp)
        self.fn = accelerator.reduce(self.fn)

    def compute(self):
        tp = self.tp.float()
        fn = self.fn.float()
        denom = tp + fn
        if denom == 0:
            recall = 0.0
        else:
            recall = tp / denom
        return {self.name: recall}
    
    def get_output(self, reduce=True):
        return self.compute()

class F1(Metric):
    def __init__(self, name="f1", device=torch.device("cpu")):
        super().__init__(name, device)
        self.tp = torch.tensor(0, device=device)
        self.fp = torch.tensor(0, device=device)
        self.fn = torch.tensor(0, device=device)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fp = torch.tensor(0, device=self.device)
        self.fn = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.fn = self.fn.to(device)

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] with 0 or 1
        y_true: ground truth (batch["prompt"]), shape [B] with 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        # True Positives
        tp_mask = (y_pred == 1) & (y_true == 1)
        # False Positives
        fp_mask = (y_pred == 1) & (y_true == 0)
        # False Negatives
        fn_mask = (y_pred == 0) & (y_true == 1)

        self.tp += tp_mask.sum()
        self.fp += fp_mask.sum()
        self.fn += fn_mask.sum()

    def sync_across_processes(self, accelerator):
        self.tp = accelerator.reduce(self.tp)
        self.fp = accelerator.reduce(self.fp)
        self.fn = accelerator.reduce(self.fn)

    def compute(self):
        """
        F1 = 2 * TP / (2*TP + FP + FN)
        """
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        numerator = 2.0 * tp
        denominator = numerator + fp + fn

        if denominator == 0:
            f1 = 0.0
        else:
            f1 = numerator / denominator

        return {self.name: f1}
    
    def get_output(self, reduce=True):
        return self.compute()
