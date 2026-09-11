from os import getenv
import subprocess


def submit(*, o, e, J, path,
          part='menon,normal,owners', a='0', c=1, mem_per_cpu='4G', t='1-0',args=None):
    """Submit a Python script if not run by a SLURM batch array.

    Used to create Python scripts that submit themselves.
    """
    if getenv('SLURM_ARRAY_TASK_ID') is None:
        if args is None:
            python_run = path
        else:
            python_run = f"{path} {' '.join(args)}"
        command = (
            'source /oak/stanford/groups/menon/projects/<user>/2025_eeg_challenge/scripts/environment.sh;'
            f'sbatch -p {part} -a {a} -n 1 '
            f'-c {c} --mem-per-cpu {mem_per_cpu} -t {t} -o {o} -e {e} -J {J} '
            f'--wrap="mprof run python {python_run}"')
        subprocess.run(command,shell=True,check=True)
        return True
    else:
        return False


def get_job(total_parallel_units):
    """Allocate unit indices to the job array member.
    
    Args:
        total_parallel_units: e.g., total number of subjects over which we are
          parallelizing.
    """
    if getenv('SLURM_ARRAY_TASK_MIN') is not None:
        assert int(getenv('SLURM_ARRAY_TASK_MIN'))==0
        cpu_index = int(getenv('SLURM_ARRAY_TASK_ID'))
        total_cpus = int(getenv('SLURM_ARRAY_TASK_MAX')) + 1
        return list(range(cpu_index, total_parallel_units, total_cpus))
    return list(range(total_parallel_units))
