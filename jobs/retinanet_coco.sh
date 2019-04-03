#!/usr/bin/env bash
#SBATCH --job-name retinanet_coco
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/maskrcnn-benchmark
#SBATCH --output ../logs/%x_%j.out

source ../venv/bin/activate
python tools/train_net.py --config-file configs/retinanet/retinanet_R-50-FPN_1x.yaml \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.005 \
    SOLVER.MAX_ITER 180000 \
    SOLVER.STEPS "(120000, 160000)" \
    TEST.IMS_PER_BATCH 4 \
    MODEL.MASK_ON False \
    OUTPUT_DIR ../work/experiments/retinanet_coco