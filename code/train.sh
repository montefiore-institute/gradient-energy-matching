#!/bin/sh
#
# Gradient Energy Matching (GEM) optimizer.
# Submission script for starting the training procedure on a single machine.

usage() {
    echo 'usage: train.sh [workers]'
    echo 'Example: ./train.sh 2'
    echo 'Please configure hyperparameters inside the submission script.'
}

# Check if the number of workers have been specified.
if [ -z "$1" ]; then
    usage
    exit 1
fi

# Initialize the job parameterization.
export BATCH_SIZE=64
export EPOCHS=1
export IFNAME="lo" # Network interface card to bind the server to.
export LEARNING_RATE=0.05
export MOMENTUM=0.9
export KAPPA=1 # Proxy amplification. Experimental results show 2 is a good value.
export MASTER="127.0.0.1:5000" # Network address of the master. Change IP in a multi-IP system. Contact author for Slurm submission scripts as they require some additional configuration.
export WORKERS=$1
export WORLD_SIZE=$(($WORKERS + 1))

# Create the folder which stores the models.
mkdir models &> /dev/null

# Start the processes in parallel.
for RANK in `seq 0 $WORKERS`; do
    export RANK=$RANK &&
    python gem.py &
done
wait # Wait for all processes to complete.
