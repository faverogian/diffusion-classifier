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
from dataset.ipmsa import LORISTransforms, IPMSADataLoader, MRIImageKeys
from diffusion.diffusion_classifier import DiffusionClassifier
from utils.metrics import Accuracy, F1, Precision, Recall
from utils.wavelet import wavelet_dec_2, wavelet_enc_2

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
    
def ipmsa_plotter(output_dir: str, batches: list, samples: list, epoch: int, process_idx: int):
    """
    Plot IPMSA samples and save them to the output_dir

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
    import numpy as np

    green_cmap = mcolors.LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'])

    slices = config.slices
    offset = slices // 2
    alpha_threshold = 0.15

    for i, (batch, sample) in enumerate(zip(batches, samples)):
        prompts = batch["prompt"]
        samples = sample

        for j in range(5): # batch size
            
            if config.wavelet_transform:
                sample_item = samples[j] * 2 # Get back to range [-1 * level, 1 * level]
                sample_item = wavelet_enc_2(sample_item) # Should be back to range [-1, 1]
            else:
                sample_item = samples[j]

            flair_pred = sample_item[offset].cpu().numpy()
            ct2f_pred = sample_item[offset+1*slices].cpu().numpy()
            
            prompt = prompts[j]
            activity = "active" if prompt else "inactive"

            fig, axs = plt.subplots(1, 1, figsize=(5, 5))

            # Plot the predicted image at W096
            ct2f_pred_alpha = ct2f_pred.copy()
            ct2f_pred_alpha[ct2f_pred_alpha <= alpha_threshold] = 0
            ct2f_pred_alpha[ct2f_pred_alpha > alpha_threshold] = 1
            axs.imshow(flair_pred, cmap='gray')
            axs.imshow(ct2f_pred, cmap=green_cmap, alpha=ct2f_pred_alpha)
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

    preprocess = transforms.Compose([
        LORISTransforms.PadTimepoints(),
        LORISTransforms.GetSlice(slices=config.slices),
        LORISTransforms.Denoise(),
        LORISTransforms.BinarizeLabel(),
        LORISTransforms.Resize(),
        LORISTransforms.ToTensor(),
        LORISTransforms.BlurLabel3D(sigma=1, kernel_size=5), # done on tensor
        LORISTransforms.NormalizeTensor(), # done on tensor
    ])

    def transform(x):
        '''
        Trials: OPERA1, OPERA2, DEFINE_ENDORSE, BRAVO
        Sequences: BEAST, FLAIR, GAD, CT2F
        Timepoints: W000, (W024), W048, W096

        Notes: 
            OPERA1, OPERA2, DEFINE_ENDORSE have W000, W024, W048, W096
            BRAVO has W000, W048, W096, W096. We will use W000, W048, W096.
            To do this, W024 is combined with W048 for OPERAs and DEFINE_ENDORSE.
        '''
        x = preprocess(x['output']) 

        # Create the target image tensor
        flair_w000 = x[MRIImageKeys.FLAIR][0]
        ct2f_w000 = x[MRIImageKeys.CT2F][0]
         
        images = torch.cat([flair_w000, ct2f_w000], dim=0)

        # Convert the tensors to half precision
        images = images.to(torch.float32)

        # Wavelet transform (one level)
        if config.wavelet_transform:
            images = wavelet_dec_2(images) / 2 # Keep in range [-1, 1]

        # Activity data
        newt2_w048 = x[MRIImageKeys.NEWT2][1]/2 + 0.5
        newt2_w096 = x[MRIImageKeys.NEWT2][2]/2 + 0.5
        newt2 = (newt2_w048 + newt2_w096).clamp(0,1)
        active_label = torch.sum(newt2) > 0

        # Make prompt for patient
        prompt = int(active_label)

        return {'images': images, 'prompt': prompt}

    train_data_path = os.path.join(config.experiment_path, 'split/train_dataset_filtered.pkl')
    val_data_path = os.path.join(config.experiment_path, 'split/val_dataset_filtered.pkl')
    test_data_path = os.path.join(config.experiment_path, 'split/test_dataset_filtered.pkl')

    ipmsa = IPMSADataLoader(
        train_data_path,
        val_data_path,
        test_data_path,
        transform,
        config.slurm,
        config.batch_size,
        config.num_workers
    )

    train_loader = ipmsa.get_train_loader()
    val_loader = ipmsa.get_val_loader()
    test_loader = ipmsa.get_test_loader()

    # Define model - somewhat aligned with simple diffusion ImageNet 128 model
    unet = UNetCondition2D(
        sample_size=config.image_size if not config.wavelet_transform else config.image_size//2,
        in_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
        out_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
        layers_per_block=(2, 2, 4, 4, 4),  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 512, 768),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
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
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        encoder_hid_dim=256,
        encoder_hid_dim_type='text_proj',
        cross_attention_dim=256
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

    metrics=[Accuracy("accuracy"), F1("f1"), Precision("precision"), Recall("recall")]

    # Train the model
    metric_output, _, _ = diffusion_classifier.inference(
                        train_dataloader=train_loader,
                        val_dataloader=test_loader,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        metrics=metrics,
                        plot_function=None,
                        classification=config.classification,
                        checkpoint_folder=config.checkpoint_folder,
                    )

    print([{k: round(v.item(), 4) for k, v in d.items()} for d in metric_output])

if __name__ == "__main__":
    main()