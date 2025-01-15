# Project path
import sys
import os
import json

# Get project root from environment variable
projectroot = os.environ['PROJECT_ROOT']
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# Project imports
from nets.dit import DiT
from dataset.chexpert import CheXpertDataLoader
from diffusion.diffusion_classifier import DiffusionClassifier
from utils.metrics import Accuracy, F1, Precision, Recall
from utils.wavelet import wavelet_dec_2, wavelet_enc_2

# Third party imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate

# Training configuration
class TrainingConfig:
    def __init__(self):
        config_str = os.environ.get('TRAINING_CONFIG')
        if config_str is None:
            raise ValueError("TRAINING_CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config['project_root']
        self.experiment_dir = self.config['experiment_dir']

        # Construct experiment path
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)
    
def chexpert_plotter(output_dir: str, batches: list, samples: list, epoch: int, process_idx: int):
    """
    Plot CheXpert samples and save them to the output_dir

    output_dir: str
        The output directory to save the plots
    batches: list
        List of batches of images
    samples: list
        List of samples
    epoch: int
        The epoch number of the training
    process_idx: int
        The process index

    Returns
        image_path: str
            The path to the saved image
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    for i, (batch, sample) in enumerate(zip(batches, samples)):
        prompts = batch["prompt"]
        samples = sample

        for j in range(1): # batch size

            if config.wavelet_transform:
                sample_item = samples[j] * 2 # [-2, 2]
                sample_item = wavelet_enc_2(sample_item)
            else:
                sample_item = samples[j]

            pred = sample_item.cpu().detach().numpy() / 2 + 0.5
            
            prompt = prompts[j]
            activity = "active" if prompt else "inactive"

            fig, axs = plt.subplots(1, 1, figsize=(5, 5))

            # Plot the predicted image at W096
            axs.imshow(pred.transpose(1, 2, 0))
            axs.axis('off')
                
            # Set top row title
            fig.suptitle(f"Patient status: {activity}", fontsize=16)
            plt.tight_layout()

            # Make path for patient
            patient_path = os.path.join(output_dir, f"{activity}")
            os.makedirs(patient_path, exist_ok=True)
            image_path = os.path.join(patient_path, f"epoch_{epoch}_sample_{j}_process_{process_idx}.png")
            plt.savefig(image_path, dpi=300)
            plt.close()

    return image_path

def main():
    global config
    config = TrainingConfig()

    # Set seed
    accelerate.utils.set_seed(config.seed)

    chexpert = CheXpertDataLoader(
        wavelet_transform=config.wavelet_transform,
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    train_loader = chexpert.get_train_loader()
    val_loader = chexpert.get_val_loader()
    test_loader = chexpert.get_test_loader()

    # Define model
    dit = DiT(
        num_attention_heads=6,
        attention_head_dim=64,
        in_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
        out_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
        num_layers=12,
        dropout=0.0,
        norm_num_groups=32,
        attention_bias=True,
        sample_size=config.image_size if not config.wavelet_transform else config.image_size//2,
        patch_size=config.patch_size,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        upcast_attention=False,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=1e-5,
    )

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(dit.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    # Create the diffusion classifier object
    diffusion_classifier = DiffusionClassifier(
        backbone=dit,
        config=config,
    )

    metrics = [
        Accuracy("accuracy"),
        F1("f1"),
        Precision("precision"),
        Recall("recall"),
    ]

    # Train the model
    diffusion_classifier.train_loop(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        checkpoint_metric="f1",
        plot_function=chexpert_plotter
    )

if __name__ == "__main__":
    main()