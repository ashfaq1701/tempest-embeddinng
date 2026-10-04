#!/usr/bin/env bash
#SBATCH --job-name=ph2_wk
#SBATCH --partition=verylong
#SBATCH --gres=gpu:RTX:1
#SBATCH --constraint=rtx
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=10-00:00:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/prevhop_fixed/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_prevhop_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL"
