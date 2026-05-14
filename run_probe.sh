#!/bin/bash
#SBATCH --job-name=slip_qd_ep10_1                # Job name
#SBATCH --output=output/slip_qd_ep10_1_%j.log    # Standard output log (%j = job ID)
#SBATCH --error=output/slip_qd_ep10_1_%j.err     # Standard error log
#SBATCH --time=2-00:00:00                        # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=1                               # 2 task — torchrun spawns one process per GPU
#SBATCH --cpus-per-task=16                       # Number of CPUs per task
#SBATCH --mem=48GB                               # 48GB model+workers + 24GB local dataset copy
#SBATCH --partition=ada                          # Partition (long/queue)
#SBATCH --gres=gpu:ADA6000:1                     # GPU allocation (if needed, modify accordingly)
#SBATCH --account=research
# #SBATCH --nodelist=cn7                         # Node to run on (modify as needed)
# =============================================================

mkdir -p output
echo "job: $SLURM_JOB_NAME"
# >>> Conda setup <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
# trap 'echo "=> cleaning up $LOCAL_DATA"; rm -rf "$LOCAL_DATA" 2>/dev/null || true' EXIT

# Job execution commands
. ./.env

python linear_probe.py \
  --root "$DATASET_ROOT" \
  --checkpoint "output/slip_vitb32_qd14_lora_ep_10_hpset_7vpoqhty_set_1/checkpoint_best.pt" \
  --gpu 0