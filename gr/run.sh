#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
pip install deepspeed==0.15.4
pip install huggingface-hub==0.33.4
pip install transformers==4.46.3
pip install accelerate==1.0.1
pip install tf-keras
pip install mpi4py

conda install -c conda-forge libopenblas openmpi libomp


bash run_train.sh