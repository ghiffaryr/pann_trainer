#!/bin/bash
PLAIN=""
FINETUNE="Transfer_"
PRETRAIN=False

CWD="/mnt/c/Ghiffary/Project/panns_trainer"

# ------ Inference audio tagging result with pretrained model. ------
WORKSPACE=$CWD"/project/cremad_32k"
HOLDOUT_FOLD=2
TASK_TYPE=$FINETUNE
MODEL_TYPE=$TASK_TYPE"Cnn14"
LOSS_TYPE="clip_balanced_bce"
AUGMENTATION="mixup"
BATCH_SIZE=32
FREEZE_BASE=True
ITERATION=2000

if [ $TASK_TYPE = $FINETUNE ]; then
    PRETRAIN=True
fi

CHECKPOINT_PATH=$WORKSPACE"/checkpoints/main/holdout_fold=$HOLDOUT_FOLD/$MODEL_TYPE/pretrain=$PRETRAIN/loss_type=$LOSS_TYPE/augmentation=$AUGMENTATION/batch_size=$BATCH_SIZE/freeze_base=$FREEZE_BASE/$ITERATION""_iterations.pth"

# Inference.
python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --sample_rate=32000 \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=80 \
    --fmax=14000 \
    --audio_path=$WORKSPACE"/dataset/Disgust/1001_IEO_DIS_MD_wav_channel1.wav" 