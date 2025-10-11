#!/usr/bin/bash

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

# python /home/jonaswil/Software/alphafold3/run_alphafold.py \
#     --run_data_pipeline=False \
#     --output_protlake tests_out/af3/af3_out_protlake \
#     --model_dir /home/jonaswil/weights/af3/model \
#     --input_dir tests/af3/testdata_af3_input \
#     --buckets 192,256,320,384,400,401,402,448,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120 > tests_out/logs/run_af3_protlake.log 2>&1

~/containers/af3_protlake/af3_protlake.sif \
    --run_data_pipeline=False \
    --output_protlake tests_out/af3/af3_out_protlake_container \
    --model_dir /home/jonaswil/weights/af3/model \
    --input_dir tests/af3/testdata_af3_input \
    --buckets 192,256,320,384,400,401,402,448,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120 > tests_out/logs/run_af3_protlake_container.log 2>&1
