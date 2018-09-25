#!/bin/bash

set -e

MIN_GPUS=4
MAX_GPUS=4
EPOCHS=100

for N_GPUS in $(seq $MIN_GPUS $MAX_GPUS)
do
	python benchmark.py train --n_gpus $N_GPUS --epochs $EPOCHS --batch_size_per_gpu 11000
done

