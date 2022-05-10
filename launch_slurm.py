import commands

def __name__ == '__main__':
    cmd = 'sbatch toy_job.sh'
    status, output = commands.getstatusoutput(cmd)