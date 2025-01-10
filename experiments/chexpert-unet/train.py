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
from nets.unet import UNetCondition2D
from dataset.chexpert import CheXpertDataLoader
from diffusion.diffusion_classifier import DiffusionClassifier
from utils.metrics import Accuracy

# Third party imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate
from torchvision import transforms

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

        for j in range(2): # batch size

            pred = samples[j].cpu().numpy()
            
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
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    train_loader = chexpert.get_train_loader()
    val_loader = chexpert.get_val_loader()
    test_loader = chexpert.get_test_loader()

    # Define model - ADM architecture
    unet = UNetCondition2D(
        sample_size=config.image_size,  # the target image resolution
        in_channels=config.image_channels,  # the number of input channels, 3 for RGB images
        out_channels=config.image_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        encoder_hid_dim=512,
        encoder_hid_dim_type='text_proj',
        cross_attention_dim=512
    )

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    # Create the diffusion classifier object
    diffusion_classifier = DiffusionClassifier(
        backbone=unet,
        config=config,
    )

    # Train the model
    diffusion_classifier.train_loop(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=[Accuracy("classification accuracy")],
        plot_function=chexpert_plotter
    )

if __name__ == "__main__":
    main()