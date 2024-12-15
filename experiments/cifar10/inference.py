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
from dataset.cifar import CIFAR10DataLoader, cifar10_classes
from diffusion.diffusion_classifier import DiffusionClassifier

# Third party imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate
from utils.metrics import Accuracy

# Inference configuration
class InferenceConfig:
    def __init__(self):
        config_str = os.environ.get('INFERENCE_CONFIG')
        if config_str is None:
            raise ValueError("INFERENCE_CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config['project_root']
        self.experiment_dir = self.config['experiment_dir']

        # Construct experiment path
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)
    
def cifar_plotter(output_dir: str, batches: list, samples: list, epoch: int, process_idx: int):
    """
    Plot CIFAR10 samples and save them to the output_dir

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
            The path to the last saved image
    """
    import matplotlib.pyplot as plt

    # Pick a random sample from "samples"
    for i in range(25):
        sample = samples[0][i].cpu().numpy()

        # Get its class
        class_id = batches[0]['prompt'][i].item()
        class_name = cifar10_classes[class_id]

        # Make a directory for the class if it doesn't exist
        if not os.path.exists(os.path.join(output_dir, f"{class_name}")):
            os.makedirs(os.path.join(output_dir, f"{class_name}"))

        # Update the image path
        image_path = os.path.join(output_dir, f"{class_name}/sample_{epoch}_{process_idx}.png")
        
        # Save the image
        plt.imsave(image_path, sample.transpose(1, 2, 0))

    return image_path
        
def main():
    global config
    config = InferenceConfig()

    # Set seed
    accelerate.utils.set_seed(config.seed)

    # Load data
    cifar10 = CIFAR10DataLoader(data_path=config.data_path, batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader = cifar10.get_train_loader()
    test_loader = cifar10.get_test_loader()

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
        encoder_hid_dim=128,
        encoder_hid_dim_type='text_proj',
        cross_attention_dim=128
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
        unet=unet,
        config=config,
    )

    # Train the model
    merics, _, _ = diffusion_classifier.inference(
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=[Accuracy("accuracy")],
        plot_function=cifar_plotter,
        classification=config.classification,
    )
    print(merics)

if __name__ == "__main__":
    main()