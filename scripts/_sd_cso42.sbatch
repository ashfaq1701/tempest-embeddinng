#!/usr/bin/env bash
#SBATCH --job-name=cso_s42
#SBATCH --partition=verylong
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=07:55:00
#SBATCH --output=/mnt/nfs2/inf/ms2420/tempest-embeddinng/logs/coso/slurm-%j.out
set -u
exec /mnt/nfs2/inf/ms2420/tempest-embeddinng/scripts/run_coso_arm.sh "$CODE" "$EXP" "$DS" "$BIP" "$TAG" "$PAT" "$SEED" "$EPOCHS" "$WPN" "$MWL" "$NL"
