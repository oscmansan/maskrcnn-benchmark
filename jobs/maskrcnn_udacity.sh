#!/usr/bin/env bash
#SBATCH --job-name maskrcnn_udacity
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/maskrcnn-benchmark
#SBATCH --output ../logs/%x_%j.out

source ../venv/bin/activate
python tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.BASE_LR 0.01 \
    SOLVER.MAX_ITER 217000 \
    SOLVER.STEPS "(145000, 193000)" \
    TEST.IMS_PER_BATCH 4 \
    MODEL.MASK_ON False \
    DATASETS.TRAIN "('udacity_train',)" \
    DATASETS.TEST "('udacity_valid', 'udacity_test')" \
    OUTPUT_DIR ../work/experiments/maskrcnn_udacity