# Standard library
import sys
import os
import json

# Get project root from environment variable
projectroot = os.environ["PROJECT_ROOT"]
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# Project imports
from dataset.ipmsa import LORISTransforms, IPMSADataLoader, MRIImageKeys
from nets.resnet import ResNet2D
from classifier.classifier import Classifier
from utils.metrics import Accuracy, Precision, Recall, F1

# Third-party imports
import torch
from torchvision import transforms
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate

# Training configuration
class TrainingConfig:
    def __init__(self):
        config_str = os.environ.get("TRAINING_CONFIG")
        if config_str is None:
            raise ValueError("TRAINING_CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config["project_root"]
        self.experiment_dir = self.config["experiment_dir"]
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)

def main():
    global config
    config = TrainingConfig()
    print(config.pretrained)

    # Set seed
    accelerate.utils.set_seed(config.seed)

    preprocess = transforms.Compose([
        LORISTransforms.PadTimepoints(),
        LORISTransforms.GetSlice(slices=config.slices),
        LORISTransforms.Denoise(),
        LORISTransforms.BinarizeLabel(),
        LORISTransforms.Resize(),
        LORISTransforms.ToTensor(),
        LORISTransforms.BlurLabel3D(sigma=1, kernel_size=5),
        LORISTransforms.NormalizeTensor(),
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

        # Activity data
        newt2_w048 = x[MRIImageKeys.NEWT2][1]/2 + 0.5
        newt2_w096 = x[MRIImageKeys.NEWT2][2]/2 + 0.5
        newt2 = (newt2_w048 + newt2_w096).clamp(0,1)
        active_label = torch.sum(newt2) > 0

        # Make prompt for patient
        prompt = int(active_label)

        return {'images': images, 'prompt': prompt}

    train_data_path = os.path.join(config.experiment_path, "split/train_dataset_filtered.pkl")
    val_data_path   = os.path.join(config.experiment_path, "split/val_dataset_filtered.pkl")
    test_data_path  = os.path.join(config.experiment_path, "split/test_dataset_filtered.pkl")

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
    val_loader   = ipmsa.get_val_loader()
    test_loader  = ipmsa.get_test_loader()

    backbone = ResNet2D(
        variant=config.variant,
        pretrained=config.pretrained,
        in_channels=config.image_channels,
    )

    classifier = Classifier(
        backbone=backbone,
        config=config
    )

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs
    )


    metrics = [Accuracy("classification accuracy"), Precision("precision"), Recall("recall"), F1("f1")]

    classifier.train_loop(
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
    )

if __name__ == "__main__":
    main()
