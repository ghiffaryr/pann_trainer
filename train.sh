#!/bin/bash
CWD="/mnt/c/Ghiffary/Project/panns_trainer"

WORKSPACE=$CWD"/project/cremad_32k"
DATASET_DIR=$WORKSPACE"/dataset"

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

HOLDOUT_FOLD=2
TASK="FINETUNE"
MODEL_TYPE="Cnn14"
PRETRAINED_MODEL=$MODEL_TYPE"_mAP=0.431.pth"
LOSS_TYPE="clip_balanced_bce"
AUGMENTATION="mixup"
BATCH_SIZE=32
FREEZE_BASE=True
LEARNING_RATE=1e-4
RESUME_ITERATION=0
STOP_ITERATION=10000


if [ $TASK = "PLAIN" ]; then
    python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --model_type=$MODEL_TYPE --loss_type=$LOSS_TYPE --augmentation=$AUGMENTATION --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --resume_iteration=$RESUME_ITERATION --stop_iteration=$STOP_ITERATION --cuda
elif [ $TASK = "FINETUNE" ] && [ $FREEZE_BASE = True ]; then
    MODEL_TYPE="Transfer_"$MODEL_TYPE
    PRETRAINED_CHECKPOINT_PATH=$CWD"/pretrained_model/"$PRETRAINED_MODEL
    python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=$LOSS_TYPE --augmentation=$AUGMENTATION --freeze_base --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --resume_iteration=$RESUME_ITERATION --stop_iteration=$STOP_ITERATION --cuda
elif [ $TASK = "FINETUNE" ] && [ $FREEZE_BASE = False ]; then
    MODEL_TYPE="Transfer_"$MODEL_TYPE
    PRETRAINED_CHECKPOINT_PATH=$CWD"/pretrained_model/"$PRETRAINED_MODEL
    python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=$LOSS_TYPE --augmentation=$AUGMENTATION --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --resume_iteration=$RESUME_ITERATION --stop_iteration=$STOP_ITERATION --cuda
fi