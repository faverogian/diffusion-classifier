from comet_ml import Experiment, ExistingExperiment
import torch
import torch.nn as nn
import os
import sys
import time
from tqdm import tqdm
from accelerate import Accelerator

class BackboneWithHead(nn.Module):
    def __init__(self, backbone, head):
        """
        A simple model that combines a backbone and a head.

        Args:
            backbone (nn.Module): A feature-extraction backbone.
            head (nn.Module): A classification head.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

class Classifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        config: dict,
    ):
        """
        A refactored Classifier that uses a backbone for feature extraction
        and a classification head on top.

        Args:
            backbone (nn.Module): Your feature-extraction backbone.
            config (dict): A config dict holding hyperparameters and other settings.
        """
        super().__init__()
        
        self.config = config

        backbone_dim = backbone.output_dim
        self.num_classes = config.classes

        self.model = BackboneWithHead(
            backbone, 
            nn.Linear(backbone_dim, self.num_classes)
        )

        # Standard cross-entropy for classification
        self.loss_fn = nn.CrossEntropyLoss()

        

    def forward(self, x):
        """
        Forward pass: extract features from backbone, then pass through classifier head.

        Args:
            x (torch.Tensor): Input tensor (e.g., images) of shape [B, C, H, W] 
                              or whatever the backbone expects.

        Returns:
            logits (torch.Tensor): Output logits of shape [B, num_classes].
        """
        return self.model(x)

    def loss(self, logits, labels):
        """
        Compute classification loss.

        Args:
            logits (torch.Tensor): Model outputs of shape [B, num_classes].
            labels (torch.Tensor): Ground-truth class labels of shape [B].

        Returns:
            torch.Tensor: A scalar loss.
        """
        return self.loss_fn(logits, labels)

    def train_one_epoch(self, model, optimizer, train_dataloader, lr_scheduler, accelerator):
        """
        Perform one training epoch over the dataset.

        Args:
            model (nn.Module): The classifier model.
            optimizer (torch.optim.Optimizer): The optimizer.
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            accelerator (accelerate.Accelerator): Accelerator for distributed/mixed-precision.

        Returns:
            (float): The average training loss for this epoch.
        """
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch in train_dataloader:
            # Example data structure: batch["images"], batch["labels"]
            x = batch["images"]
            labels = batch["prompt"]

            with accelerator.accumulate(model):
                logits = model(x)
                loss_value = self.loss(logits, labels)

                accelerator.backward(loss_value)

                if accelerator.sync_gradients:
                    model_params = dict(model.named_parameters())
                    all_params = {**model_params}
                    accelerator.clip_grad_norm_(all_params.values(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss_value.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, model, val_dataloader, stop_idx, metrics=None):
        """
        Evaluate the model on a validation dataloader.

        Args:
            model (nn.Module): The classifier model.
            val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
            accelerator (accelerate.Accelerator): Accelerator for distributed/mixed-precision.
            metrics (list, optional): A list of metric objects to update. Each metric must have
                                      .update((predictions, batch)) and .get_output() methods.

        Returns:
            (float, dict or None):
                - avg_loss: The average validation loss
                - metrics_output: A dict of all metrics results if metrics are provided, else None
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(val_dataloader, desc="Evaluating")

        for idx, batch in enumerate(val_dataloader):
            progress_bar.update(1)
            x = batch["images"]
            labels = batch["prompt"]

            logits = model(x)
            loss_value = self.loss(logits, labels)

            total_loss += loss_value.item() * x.size(0)
            total_samples += x.size(0)

            if metrics is not None:
                preds = torch.argmax(logits, dim=1)
                for metric in metrics:
                    metric.update((preds, batch))
            if stop_idx is not None and idx == stop_idx:
                break

        progress_bar.close()

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, metrics

    def train_loop(
        self, 
        optimizer, 
        train_dataloader, 
        val_dataloader, 
        lr_scheduler,
        metrics=None,
    ):
        """
        The main multi-epoch training loop, analogous to the original `train_loop`.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.
            val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            metrics (list, optional): A list of metrics to compute/evaluate.
            plot_function (callable, optional): A custom function to visualize or log samples.

        Returns:
            None
        """
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=self.config.experiment_path,
        )

        # Wrap model, optimizer, dataloaders, scheduler with accelerator
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        if metrics is not None:
            for metric in metrics:
                metric.set_device(accelerator.device)

        # Setup Comet experiment (if desired)
        experiment = None
        start_epoch = 0

        if self.config.resume:
            checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
            start_epoch, experiment_key = self.load_checkpoint(checkpoint_path, accelerator)

            if experiment_key is not None and self.config.use_comet and accelerator.is_main_process:
                experiment = ExistingExperiment(
                    previous_experiment=experiment_key,
                    api_key=self.config.comet_api_key,
                )
        else:
            if self.config.use_comet and accelerator.is_main_process:
                experiment = Experiment(
                    api_key=self.config.comet_api_key,
                    project_name=self.config.comet_project_name,
                    workspace=self.config.comet_workspace,
                )
                experiment.set_name(self.config.comet_experiment_name)
                experiment.log_asset(os.path.join(self.config.experiment_path, 'train.py'), 'train.py')
                experiment.log_asset(os.path.join(self.config.project_root, 'train-classifier.sh'), 'train-classifier.sh')
                experiment.log_other("GPU Model", torch.cuda.get_device_name(0))
                experiment.log_other("Python Version", sys.version)

        # Train!
        
        if accelerator.is_main_process:
            print(f"Config:\n{self.config.__dict__}")

        num_epochs = self.config.num_epochs
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            # One training epoch
            avg_train_loss = self.train_one_epoch(model, optimizer, train_dataloader, lr_scheduler, accelerator)

            epoch_elapsed = time.time() - epoch_start_time
            if accelerator.is_main_process:
                print(f"Epoch {epoch}/{num_epochs - 1}, "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Time: {epoch_elapsed:.2f}s")

                if experiment is not None:
                    experiment.log_metric("train_loss", avg_train_loss, epoch=epoch)

            # Validation & checkpointing
            if epoch % self.config.eval_period == 0 or epoch == num_epochs - 1:
                val_evaluation_start_time = time.time()
                val_loss, val_metrics = self.evaluate(
                    model,
                    val_dataloader, 
                    stop_idx=self.config.evaluation_batches,
                    metrics=metrics)

                if val_metrics is not None:
                    for metric in val_metrics:
                        metric.sync_across_processes(accelerator)
                        metric_output = metric.get_output()
                        if experiment is not None and accelerator.is_main_process:
                            metric_output = {f"val_{metric_name}": value for metric_name, value in metric_output.items()}
                            experiment.log_metrics(metric_output, step=epoch)
                        base_line_accuracy = 1/self.config.n_fast_classes if self.config.fast_classification else 1/self.config.classes
                        if accelerator.is_main_process:
                            print(f"Baseline Classification Accuracy: {base_line_accuracy:.2f}")
                            print(metric_output)
                        metric.reset()

                val_evaluation_elapsed = time.time() - val_evaluation_start_time
                if accelerator.is_main_process:
                    if experiment is not None:
                        experiment.log_metric("val_loss", val_loss, epoch=epoch)

                    self.save_checkpoint(accelerator, epoch, val_metrics, experiment)
                    print(f"Val evaluation time: {val_evaluation_elapsed:.2f} s.")

    @torch.no_grad()
    def inference(self, optimizer, train_dataloader, val_dataloader, lr_scheduler, metrics=None):
        """
        An inference function that loads the latest checkpoint and evaluates on val_dataloader.

        Args:
            optimizer (torch.optim.Optimizer): Typically unused here, but kept for consistency.
            train_dataloader (torch.utils.data.DataLoader): Typically unused, can remove if not needed.
            val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Typically unused, can remove if not needed.
            metrics (list, optional): A list of metric objects for evaluation.

        Returns:
            (dict): Final metrics on the validation set if any are computed.
        """
        accelerator = Accelerator()

        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # Load checkpoint
        checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
        self.load_checkpoint(checkpoint_path, accelerator)

        if metrics is not None:
            for metric in metrics:
                metric.set_device(accelerator.device)

        val_loss, val_metrics = self.evaluate(
            model, 
            val_dataloader, 
            metrics=metrics,
            stop_idx=self.config.evaluation_batches
        )

        metric_output = []
        if metrics is not None:
            for metric in metrics:
                metric.sync_across_processes(accelerator)
                metric_output.append(metric.get_output())

        return val_loss, metric_output

    @torch.no_grad()
    def classify(self, x):
        """
        Simple classify method: forward pass and argmax over the logits.

        Args:
            x (torch.Tensor): Input image batch of shape [B, C, H, W].

        Returns:
            torch.Tensor: Predicted class indices of shape [B].
        """
        self.model.eval()
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def save_checkpoint(self, accelerator: Accelerator, epoch, metrics, experiment):
        """
        Saves the model checkpoint.

        Args:
            accelerator (accelerate.Accelerator): The Accelerator instance.
            epoch (int): Current epoch.
            metrics (dict): Latest metrics dict.
            experiment (comet_ml.Experiment): The Comet experiment, if any.
        """
        checkpoint_dir = os.path.join(self.config.experiment_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Accelerator state
        accelerator.save_state(output_dir=checkpoint_dir)

        # Save experiment state
        experiment_key = experiment.get_key() if experiment is not None else None
        experiment_state = {
            'epoch': epoch + 1,
            'experiment_key': experiment_key
        }

        path = os.path.join(checkpoint_dir, "experiment_state.pth")
        torch.save(experiment_state, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path, accelerator: Accelerator):
        """
        Loads the model checkpoint from disk.

        Args:
            checkpoint_path (str): Path to the checkpoint folder.
            accelerator (accelerate.Accelerator): The Accelerator instance.

        Returns:
            (int, str or None): The epoch to resume from, and the experiment key (if any).
        """
        if not os.path.exists(checkpoint_path):
            print("No checkpoint directory found. Starting from scratch.")
            return 0, None

        # Restore accelerator state
        accelerator.load_state(input_dir=checkpoint_path)

        state_file = os.path.join(checkpoint_path, "experiment_state.pth")
        if not os.path.isfile(state_file):
            print("No experiment_state.pth found. Starting from scratch.")
            return 0, None

        # Load the checkpoint
        checkpoint = torch.load(state_file, map_location='cpu', weights_only=False)
        epoch = checkpoint['epoch']
        experiment_key = checkpoint['experiment_key']

        print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
        return epoch, experiment_key
