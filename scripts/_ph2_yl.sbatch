#!/usr/bin/env bash
#SBATCH --job-name=ph2_yl
#SBATCH --partition=long
#SBATCH --gres=gpu:RTX:1
#SBATCH --constraint=rtx
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/prevhop_fixed/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_prevhop_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL"
