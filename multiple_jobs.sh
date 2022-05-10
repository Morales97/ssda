#! /bin/bash
# from https://hpc.nih.gov/docs/job_dependencies.html
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1

# first job - no dependencies
jid1=$(sbatch toy_job.sh)

# multiple jobs can depend on a single job
jid2=$(sbatch  --dependency=afterany:$jid1 toy_job2.sh)
