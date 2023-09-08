#!/bin/bash -l
#SBATCH --job-name=dialogpt_finetune
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=ngpu[02]
#SBATCH -o train_output_1

# General SLURM Parameters
echo "# SLURM_JOBID  = ${SLURM_JOBID}"
echo "# SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "# SLURM_NNODES = ${SLURM_NNODES}"
echo "# SLURM_NTASKS = ${SLURM_NTASKS}"
echo "# SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "# SLURMTMPDIR = ${SLURMTMPDIR}"
echo "# Submission directory = ${SLURM_SUBMIT_DIR}"

# Modules
module purge

# PYTHON
PYTHON="/userspace/tdv/miniconda3/bin/python"

# Versions
echo "cuda: " $(which cuda)
echo "python: " $(which ${PYTHON})
echo "python version: $(${PYTHON} --version)"

# Path to file
TASK="finetuning.py"
REQUIREMENTS_PATH="requirements.txt"

# CMD
CMD="${PYTHON} -m pip install -r ${REQUIREMENTS_PATH} && ${PYTHON} ${TASK}"

# Launch
${CMD}