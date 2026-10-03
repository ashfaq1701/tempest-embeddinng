#!/usr/bin/env bash
#SBATCH --job-name=tho_ml
#SBATCH --partition=verylong
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=10-00:00:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/thetao/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_thetao_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL" "$NL"
