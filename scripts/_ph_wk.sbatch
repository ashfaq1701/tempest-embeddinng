#!/usr/bin/env bash
#SBATCH --job-name=ph_wk
#SBATCH --partition=long
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/prevhop/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_prevhop_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL"
