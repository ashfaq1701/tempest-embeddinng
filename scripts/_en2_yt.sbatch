#!/usr/bin/env bash
#SBATCH --job-name=en2_yt
#SBATCH --partition=general
#SBATCH --gres=gpu:RTX:1
#SBATCH --constraint=rtx
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=07:55:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/walkemanl2/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_walkemanl2_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL"
