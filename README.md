# CUDA-Based Parallel N-Body Simulation





# Build instructions

Run the following commands in the project directory:



        make clean

        make

    

This creates two executables: nbody (CPU sequential) and nbodygpu (GPU CUDA).

Note: This requires the CUDA toolkit (nvcc) to be loaded on the system (e.g., module load cuda/12.4).



# Usage

Next run the following command with the desired simulation parameters.

For the GPU version:


        ./nbodygpu <input> <dt> <nbstep> <printevery> <blocksize>


For the CPU version:

        ./nbody <input> <dt> <nbstep> <printevery>


<input>: "planet" for solar system, a number (for random particle count), or a filename to load a state from.


<dt>: The time step size (e.g., 0.01).


<nbstep>: The total number of simulation steps (e.g., 1000).


<printevery>: The frequency to print the simulation state (e.g., 100).


<blocksize>: The CUDA block size (e.g., 128).


For example, to run the solar system simulation on the GPU with a block size of 128:

        ./nbodygpu planet 0.01 1000 100 128




# Clean

To remove the compiled executable run the following command:



        make clean



# BenchMark

When Benchmarking on Centaurus:



        cd nbodygpu

        chmod +x queue_main.sh

        sbatch queue_main.sh
    
    
This executes the batch script, which will compile the code and run the benchmarks defined in queue_main.sh.

It creates the file results.txt, which will contain the output and runtimes comparing the CPU and GPU implementations across different particle counts (1000, 10000, 100000).
