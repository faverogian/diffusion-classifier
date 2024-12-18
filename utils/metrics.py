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