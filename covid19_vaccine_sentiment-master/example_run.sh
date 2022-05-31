#!/bin/bash
#SBATCH -A <allocation>               # Allocation
#SBATCH -p <queue>                # Queue
#SBATCH -t <walltime>             # Walltime/duration of the job

# set date
date="2021-02-14"

jid0=($(sbatch --account=<allocation> --partition=<queue> --time=<walltime> --nodes=1 --ntasks-per-node=1 --mem=8G --job-name="${date}"_cpu_part --output=job_"${date}"_cpu_part1.out cpu_part1.sh))

echo "jid0 ${jid0[-1]}" >> slurm_ids

jid1=($(sbatch --dependency=afterok:${jid0[-1]} --account=<allocation> --partition=<queue> --time=<walltime> --gres=gpu:a100:1 --nodes=1 --ntasks-per-node=1 --mem=32G --job-name="${date}"_gpu_part --output=job_"${date}"_gpu_part.out --export=DEPENDENTJOB=${jid0[-1]} gpu_part.sh))
 
echo "jid1 ${jid1[-1]}" >> slurm_ids

jid2=($(sbatch --dependency=afterok:${jid1[-1]} --account=<allocation> --partition=<queue> --time=<walltime> --nodes=1 --ntasks-per-node=1 --mem=8G --job-name="${date}"_cpu_part --output=job_"${date}"_cpu_part2.out --export=DEPENDENTJOB=${jid1[-1]} cpu_part2.sh))
echo "jid2 ${jid2[-1]}" >> slurm_ids
