#!/usr/bin/env bash
#SBATCH --job-name=en2_yl
#SBATCH --partition=verylong
#SBATCH --gres=gpu:RTX:1
#SBATCH --constraint=rtx
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=10-00:00:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/walkemanl2/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_walkemanl2_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL"
