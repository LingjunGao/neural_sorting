#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCENE_DIR="$EXAMPLES_DIR/data/360_v2"

DEFAULT_CKPT_ROOT="$EXAMPLES_DIR/results/mlp_checkpoint"
NEW_CKPT_ROOT="$EXAMPLES_DIR/results/mlp_checkpoint_new"

if find "$NEW_CKPT_ROOT" -type f -path "*/ckpts/ckpt_*_rank0.pt" 2>/dev/null | head -n 1 | grep -q .; then
    CKPT_ROOT="$NEW_CKPT_ROOT"
    echo "Using fine-tuned checkpoints: $CKPT_ROOT"
else
    CKPT_ROOT="$DEFAULT_CKPT_ROOT"
    echo "mlp_checkpoint_new not found (or empty). Using default checkpoints: $CKPT_ROOT"
fi

# Write current run outputs to a separate folder.
RESULT_DIR="$EXAMPLES_DIR/results"
TYPE="mlp-nonclone"
SCENE_LIST="bicycle bonsai counter kitchen room"
RENDER_TRAJ_PATH="ellipse"
FAILED_SCENES=()
DONE_SCENES=()

# Associative arrays for best-PSNR tracking per scene.
declare -A BEST_PSNR
declare -A BEST_CKPT

for SCENE in $SCENE_LIST;
do
    SCENE_FAILED=0

    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4 #4
    fi
    
    echo "Running $SCENE"

    # #  train without eval
    # CUDA_VISIBLE_DEVICES=3 python load_simple_trainer.py default --eval_steps -1 --disable_viewer  --type $TYPE --data_factor $DATA_FACTOR \
    #     --render_traj_path $RENDER_TRAJ_PATH \
    #     --data_dir data/360_v2/$SCENE/ \
    #     --result_dir $RESULT_DIR/$SCENE/$TYPE

    # run eval and render: load checkpoints from results/benchmark
    # support both:
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

    for CKPT in "${CKPTS[@]}";
    do
        OUTPUT=$(CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python "$EXAMPLES_DIR/load_simple_trainer.py" default --disable_viewer --type "$TYPE" --data_factor "$DATA_FACTOR" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "$SCENE_DIR/$SCENE/" \
            --result_dir "$RESULT_DIR/$TYPE/$SCENE" \
            --ckpt "$CKPT" 2>&1) || {
            echo "$OUTPUT"
            echo "[WARN] Evaluation failed for scene=$SCENE, ckpt=$CKPT"
            echo "[WARN] Skipping remaining checkpoints for $SCENE and continuing to next scene."
            SCENE_FAILED=1
            break
        }
        echo "$OUTPUT"

        # Extract PSNR value from output line like "PSNR: 27.123, SSIM: ..."
        CUR_PSNR=$(echo "$OUTPUT" | grep -oP 'PSNR:\s*\K[0-9]+\.?[0-9]*' | tail -n 1)
        if [[ -n "$CUR_PSNR" ]]; then
            if [[ -z "${BEST_PSNR[$SCENE]:-}" ]] || (( $(echo "$CUR_PSNR > ${BEST_PSNR[$SCENE]}" | bc -l) )); then
                BEST_PSNR[$SCENE]="$CUR_PSNR"
                BEST_CKPT[$SCENE]="$CKPT"
            fi
        fi
    done

    if [ "$SCENE_FAILED" -eq 1 ]; then
        FAILED_SCENES+=("$SCENE")
        continue
    fi

    DONE_SCENES+=("$SCENE")
done

echo "============================================================"
echo "Evaluation finished."
echo "Successful scenes: ${DONE_SCENES[*]:-none}"
echo "Failed scenes: ${FAILED_SCENES[*]:-none}"
echo ""
printf "%-15s %-12s %s\n" "Scene" "Best PSNR" "Checkpoint"
printf "%-15s %-12s %s\n" "-----" "---------" "----------"
for SCENE in $SCENE_LIST; do
    if [[ -n "${BEST_PSNR[$SCENE]:-}" ]]; then
        printf "%-15s %-12s %s\n" "$SCENE" "${BEST_PSNR[$SCENE]}" "${BEST_CKPT[$SCENE]}"
    else
        printf "%-15s %-12s %s\n" "$SCENE" "N/A" "N/A"
    fi
done
