#!/bin/bash
#SBATCH --job-name=slip_qd_ep10_1                    # Job name
#SBATCH --output=output/slip_qd_ep10_1_%j.log        # Standard output log (%j = job ID)
#SBATCH --error=output/slip_qd_ep10_1_%j.err         # Standard error log
#SBATCH --time=2-00:00:00                     # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=1                            # 2 task — torchrun spawns one process per GPU
#SBATCH --cpus-per-task=16                     # Number of CPUs per task
#SBATCH --mem=96GB                            # 48GB model+workers + 24GB local dataset copy
#SBATCH --partition=ada                       # Partition (long/queue)
#SBATCH --gres=gpu:ADA6000:2                  # GPU allocation (if needed, modify accordingly)
#SBATCH --account=research
#SBATCH --nodelist=cn7                        # Node to run on (modify as needed)
# =============================================================

echo "job: $SLURM_JOB_NAME"
# >>> Conda setup <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
# trap 'echo "=> cleaning up $LOCAL_DATA"; rm -rf "$LOCAL_DATA" 2>/dev/null || true' EXIT

# Job execution commands
. ./.env
echo $DATASET_ROOT
echo $SLURM_JOBID

# 1) Find a free port by binding to port 0
export MASTER_PORT=$(python - <<'EOF'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
EOF
)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF
export SLURM_NNODES=${SLURM_NNODES:-1}                                                                                                                                                                                                                       
export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}
echo "nnodes: $SLURM_NNODES"
echo "nproc_per_node: $SLURM_GPUS_ON_NODE"
echo "master port: $MASTER_PORT"

LOCAL_DATA="$DATASET_ROOT"
# LOCAL_DATA=/tmp/quickdraw_$SLURM_JOBID
# if [ ! -d "$LOCAL_DATA" ]; then
#     echo "=> copying dataset to local disk ($LOCAL_DATA)..."
#     mkdir -p "$LOCAL_DATA"
#     rsync -av --include="*.ptr.npy" --include="*.strokes.npy" --exclude="*" "$DATASET_ROOT"/ "$LOCAL_DATA"/ \
#         || { echo "rsync failed (exit $?), falling back to NFS path"; LOCAL_DATA="$DATASET_ROOT"; }
#     echo "=> done: $(du -sh $LOCAL_DATA | cut -f1)"
# fi

export WANDB_DIR=/tmp

python -u -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port $MASTER_PORT \
    main.py \
  --model "SLIP_VITB32" \
  --dataset "quickdraw" \
  --root "$LOCAL_DATA" \
  --print-freq 100 \
  --workers 6 \
  --use-lora \
  --lora-rank 16 \
  --lora-alpha 16 \
  --ssl-scale 0.5 \
  --ssl-temp 0.05 \
  --epochs 10 \
  --batch-size 512 \
  --update-freq 2 \
  --lr 1e-3 \
  --output-dir "./output/slip_vitb32_qd14_lora_ep_10_hpset_7vpoqhty_set_1" \
  --wandb \
  --profile
# keep bs>2
# update-freq does gradient accumulation, so effective batch size = batch-size * update-freq
# set:
#   closed : 0
#   open : 1,2,3 for idx-1 respectively