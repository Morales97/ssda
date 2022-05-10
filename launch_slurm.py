import subprocess

if __name__ == '__main__':
    print('running launch_slurm.py...')
    cmd = 'sbatch toy_job.sh'
    status, output = subprocess.getstatusoutput(cmd)