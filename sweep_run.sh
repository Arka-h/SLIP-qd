#!/bin/bash
# Called by: wandb agent <sweep_id>
# Receives swept args as: --lr 3e-4 --lora_rank 8 --ssl_scale 0.5 --ssl_temp 0.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
. ./.env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$(python - <<'EOF'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
EOF
)

# Convert wandb-style args (--foo_bar=val) to argparse-style (--foo-bar val)
# and extract lora_rank so we can set lora_alpha = lora_rank
LORA_RANK=16
converted_args=()
for arg in "$@"; do
    if [[ "$arg" == --*=* ]]; then
        name="${arg%%=*}"
        value="${arg#*=}"
        name="${name//_/-}"
        [[ "$name" == "--lora-rank" ]] && LORA_RANK="$value"
        converted_args+=("$name" "$value")
    elif [[ "$arg" == --* ]]; then
        converted_args+=("${arg//_/-}")
    else
        converted_args+=("$arg")
    fi
done

trap '' INT
python -u -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port $MASTER_PORT \
    main.py \
    --model "SLIP_VITB32" \
    --dataset "quickdraw" \
    --root "$DATASET_ROOT" \
    --workers 3 \
    --use-lora \
    --lora-alpha $LORA_RANK \
    --epochs 5 \
    --batch-size 512 \
    --update-freq 2 \
    --print-freq 100 \
    --output-dir "./output/sweep/$WANDB_RUN_ID" \
    --wandb \
    "${converted_args[@]}"
