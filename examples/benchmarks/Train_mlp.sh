#!/bin/bash
set -euo pipefail

EXAMPLES_DIR="/mnt/ccnas2/bdp/lg524/neural_sorting/examples"
SCENE_DIR="$EXAMPLES_DIR/data/360_v2"
CKPT_ROOT="$EXAMPLES_DIR/results/benchmark"

# Dedicated output root for this script.
RESULT_ROOT="$EXAMPLES_DIR/results/mlp_checkpoint"
TYPE="mlp-nonclone"
SCENE_LIST="kitchen garden bicycle bonsai counter room"
RENDER_TRAJ_PATH="ellipse"
GPU_ID=0

for SCENE in $SCENE_LIST; do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "============================================================"
    echo "Scene: $SCENE"

    # Support both layouts:
    #   results/benchmark/<scene>/ckpts/ckpt_*_rank0.pt
    #   results/benchmark/<scene>/<exp>/ckpts/ckpt_*_rank0.pt
    mapfile -t CKPTS < <(
        {
            find "$CKPT_ROOT" -type f -path "*/$SCENE/ckpts/ckpt_*_rank0.pt"
            find "$CKPT_ROOT" -type f -path "*/$SCENE/*/ckpts/ckpt_*_rank0.pt"
        } | sort -V | uniq
    )

    if [ ${#CKPTS[@]} -eq 0 ]; then
        echo "No checkpoint found for scene=$SCENE under $CKPT_ROOT, skipping."
        continue
    fi

    for CKPT in "${CKPTS[@]}"; do
        REL_PATH="${CKPT#$CKPT_ROOT/}"
        RUN_TAG="${REL_PATH//\//_}"
        RUN_TAG="${RUN_TAG%.pt}"

        RESULT_DIR="$RESULT_ROOT/$SCENE/$RUN_TAG"

        echo "Train from: $CKPT"
        echo "Result dir: $RESULT_DIR"

        CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$GPU_ID python "$EXAMPLES_DIR/load_simple_trainer.py" default \
            --disable_viewer \
            --type "$TYPE" \
            --data_factor "$DATA_FACTOR" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "$SCENE_DIR/$SCENE/" \
            --result_dir "$RESULT_DIR" \
            --ckpt "$CKPT"
    done
done
