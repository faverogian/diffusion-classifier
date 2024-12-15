export PROJECT_ROOT="/cim/faverog/diffusion-classifier"                             
export EXPERIMENT_DIR="/experiments/cifar10"
export DATA_PATH="/cim/data/CIFAR10"

export IMAGE_SIZE=32                    # (int) Size of the input images          
export IMAGE_CHANNELS=3                 # (int) Number of channels in the input images
export BATCH_SIZE=128                   # (int) Batch size for training
export NUM_EPOCHS=500                   # (int) Number of epochs to train for
export GRADIENT_ACCUMULATION_STEPS=1    # (int) Number of gradient accumulation steps
export LEARNING_RATE=0.0001             # (float) Learning rate
export LR_WARMUP_STEPS=250              # (int) Number of warmup steps for the learning rate
export EVALUATION_BATCHES=4             # (int) Number of batches to evaluate on

export PRED_PARAM="v"                   # (str) Diffusion parameterization ('v' or 'eps')
export SCHEDULE="cosine"                # (str) Learning rate schedule ('cosine', 'shifted_cosine')
export NOISE_D=64                       # (int) Reference noise dimensionality (simple diffusion, Hoogeboom et al. 2023)
export MIXED_PRECISION="fp16"           # (str) Mixed precision training ('fp16' or 'fp32' or 'none')
export NUM_WORKERS=24                   # (int) Number of workers for the data loader

export CLASSES=10                       # (int) Number of classes in the dataset
export ENCODER_TYPE="nn"                # (str) Type of encoder for the end-to-end model ('nn' or 't5')
export CFG_W=7                          # (int) Classifier guidance scale

export SAMPLING_STEPS=50                # (int) Number of sampling steps for the reverse diffusion process

export SEED=42
export USE_COMET=0
export COMET_PROJECT_NAME=""            # (str) Comet project name
export COMET_WORKSPACE=""               # (str) Comet workspace
export COMET_EXPERIMENT_NAME="cifar10"  # (str) Comet experiment name
export COMET_API_KEY=""                 # (str) Comet API key

export CLASSIFICATION=true              # (bool) Whether to perform classification or not
export N_STAGES=2                       # (int) Number of stages for the classification
export EVALUATION_PER_STAGE=[50,500]    # (list) Number of samples to evaluate per stage
export N_KEEP_PER_STAGE=[5,1]           # (list) Number of classes to keep per stage (Must end with 1)

export INFERENCE_CONFIG="{
  \"project_root\": \"$PROJECT_ROOT\",
  \"experiment_dir\": \"$EXPERIMENT_DIR\",
  \"data_path\": \"$DATA_PATH\",
  \"image_size\": $IMAGE_SIZE,
  \"image_channels\": $IMAGE_CHANNELS,
  \"batch_size\": $BATCH_SIZE,
  \"num_epochs\": $NUM_EPOCHS,
  \"gradient_accumulation_steps\": $GRADIENT_ACCUMULATION_STEPS,
  \"learning_rate\": $LEARNING_RATE,
  \"lr_warmup_steps\": $LR_WARMUP_STEPS,
  \"evaluation_batches\": $EVALUATION_BATCHES,
  \"pred_param\": \"$PRED_PARAM\",
  \"schedule\": \"$SCHEDULE\",
  \"noise_d\": $NOISE_D,
  \"mixed_precision\": \"$MIXED_PRECISION\",
  \"num_workers\": $NUM_WORKERS,
  \"classes\": $CLASSES,
  \"encoder_type\": \"$ENCODER_TYPE\",
  \"cfg_w\": $CFG_W,
  \"sampling_steps\": $SAMPLING_STEPS,
  \"seed\": $SEED,
  \"use_comet\": $USE_COMET,
  \"comet_project_name\": \"$COMET_PROJECT_NAME\",
  \"comet_workspace\": \"$COMET_WORKSPACE\",
  \"comet_experiment_name\": \"$COMET_EXPERIMENT_NAME\",
  \"comet_api_key\": \"$COMET_API_KEY\",
  \"classification\": $CLASSIFICATION,
  \"n_stages\": $N_STAGES,
  \"evaluation_per_stage\": $EVALUATION_PER_STAGE,
  \"n_keep_per_stage\": $N_KEEP_PER_STAGE
}"

# Run the Python script
port=$(shuf -i 1025-65535 -n 1)
accelerate launch --multi-gpu \
		              --main-process-port=$port \
                  --num-machines=1 \
                  --num-processes=4 \
                  --mixed_precision='fp16' \
                  --gpu_ids=0,1,2,4\
                  $PROJECT_ROOT$EXPERIMENT_DIR/inference.py