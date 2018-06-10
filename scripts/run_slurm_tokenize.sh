#!/bin/bash
#SBATCH --partition=jag --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20480
#SBATCH --gres=gpu:1
#SBATCH --job-name="tokenize"
#SBATCH --output=/sailhome/pengqi/logs/slurm-%j.out
#SBATCH --mail-user=pengqi@cs.stanford.edu
#SBATCH --mail-type=FAIL
#SBATCH --exclude=jagupard15

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

tb=$1
shift
root=$1
shift
args=$@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
export PATH=/usr/local/cuda-8.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-8.0
cd $root
export PATH=/juicier/scr127/scr/pengqi/anaconda3_slurm/bin:$PATH
export LD_LIBRARY_PATH=/juicier/scr127/scr/pengqi/anaconda3_slurm/lib:$LD_LIBRARY_PATH

bash scripts/run_tokenize.sh $tb $CUDA_VISIBLE_DEVICES $args

#
echo "Done"
