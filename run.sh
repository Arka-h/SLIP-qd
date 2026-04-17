#!/bin/bash
#SBATCH --job-name=slip_qd_1                    # Job name
#SBATCH --output=output/slip_qd_1_%j.log        # Standard output log (%j = job ID)
#SBATCH --error=output/slip_qd_1_%j.err         # Standard error log
#SBATCH --time=2-00:00:00                     # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=2                            # Number of tasks (typically 1 for single-node jobs
#SBATCH --cpus-per-task=8                     # Number of CPUs per task
#SBATCH --mem=48GB                            # Memory allocation
#SBATCH --partition=ada                       # Partition (long/queue)
#SBATCH --gres=gpu:ADA6000:2                  # GPU allocation (if needed, modify accordingly)
#SBATCH --account=research
# #SBATCH --nodelist=cn8                        # Node to run on (modify as needed)
# =============================================================

echo "job: $SLURM_JOB_NAME"
# >>> Conda setup <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_NNODES=${SLURM_NNODES:-1}                                                                                                                                                                                                                       
export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}
echo "nnodes: $SLURM_NNODES"
echo "nproc_per_node: $SLURM_GPUS_ON_NODE"
echo "master port: $MASTER_PORT"

python -u -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port $MASTER_PORT \
    main.py \
  --model "SLIP_VITB32" \
  --dataset "quickdraw" \
  --root $DATASET_ROOT \
  --print-freq 100 \
  --workers 2 \
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