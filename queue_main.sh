#!/bin/bash
#SBATCH --job-name=NBodyBench
#SBATCH --partition=GPU
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -e

module load cuda/12.4

cd $SLURM_SUBMIT_DIR

make clean
make

RESULTS_FILE="results.txt"
rm -f $RESULTS_FILE

echo "N-Body Simulation Benchmark Results (CPU vs GPU)" > $RESULTS_FILE
echo "------------------------------------------------" >> $RESULTS_FILE

STEPS=10
DT=0.01
BLOCKSIZE=128

for N in 1000 10000 100000; do
  echo "--- Testing with N=$N particles ($STEPS steps) ---" | tee -a $RESULTS_FILE
  
  echo "Running CPU (Sequential)..."
  echo "CPU (N=$N):" >> $RESULTS_FILE
  (time ./nbody $N $DT $STEPS $((STEPS+1)) > /dev/null) 2>> $RESULTS_FILE
  
  echo "Running GPU (CUDA)..."
  echo "GPU (N=$N):" >> $RESULTS_FILE
  (time ./nbodygpu $N $DT $STEPS $((STEPS+1)) $BLOCKSIZE > /dev/null) 2>> $RESULTS_FILE
  
  echo "" >> $RESULTS_FILE
done

echo "Benchmark complete. Check $RESULTS_FILE."
