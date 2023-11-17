#!/bin/bash

function set_param_7b {
    num_gpus=1
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
    num_layers=32
    num_heads=32
    num_kv_heads=${num_heads}
    hidden_size=4096
    inter_size=11008
    vocab_size=32000
    n_positions=4096
}

function set_param_13b {
    num_gpus=1
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
    num_layers=40
    num_heads=40
    num_kv_heads=${num_heads}
    hidden_size=5120
    inter_size=13824
    vocab_size=32000
    n_positions=4096
}

function set_param_30b {
    num_gpus=2
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
    num_layers=62
    num_heads=52
    num_kv_heads=${num_heads}
    hidden_size=6656
    inter_size=17920
    vocab_size=32000
    n_positions=4096
}

function set_param_70b_tp1_pp2 {
    num_gpus=2
    pp_size=2
    tp_size=$(( num_gpus / pp_size ))
    num_layers=80
    num_heads=64
    num_kv_heads=8
    hidden_size=8192
    inter_size=28672
    vocab_size=32000
    n_positions=4096
}

function set_param_70b_tp1_pp4 {
    num_gpus=4
    pp_size=4
    tp_size=$(( num_gpus / pp_size ))
    num_layers=80
    num_heads=64
    num_kv_heads=8
    hidden_size=8192
    inter_size=28672
    vocab_size=32000
    n_positions=4096
}

function set_param_70b_tp2_pp1 {
    num_gpus=2
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
    num_layers=80
    num_heads=64
    num_kv_heads=8
    hidden_size=8192
    inter_size=28672
    vocab_size=32000
    n_positions=4096
}

function build_engine_fp16_ifb {
    python ${folder_tekit}/examples/llama/build.py \
        --output_dir ${engine_dir} \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_rmsnorm_plugin float16 \
        --enable_context_fmha \
        --remove_input_padding \
        --use_inflight_batching \
        --parallel_build \
        --n_positions ${n_positions} \
        --max_batch_size ${max_batch_size} \
        --n_layer ${num_layers} \
        --n_head ${num_heads} \
        --n_kv_head ${num_kv_heads} \
        --n_embd ${hidden_size} \
        --inter_size ${inter_size} \
        --vocab_size ${vocab_size} \
        --world_size ${num_gpus} \
        --tp_size ${tp_size} \
        --pp_size ${pp_size}
}

function build_engine_int8_ifb {
    python ${folder_tekit}/examples/llama/build.py \
        --output_dir ${engine_dir} \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_rmsnorm_plugin float16 \
        --use_weight_only \
        --enable_context_fmha \
        --remove_input_padding \
        --use_inflight_batching \
        --parallel_build \
        --n_positions ${n_positions} \
        --max_batch_size ${max_batch_size} \
        --n_layer ${num_layers} \
        --n_head ${num_heads} \
        --n_kv_head ${num_kv_heads} \
        --n_embd ${hidden_size} \
        --inter_size ${inter_size} \
        --vocab_size ${vocab_size} \
        --world_size ${num_gpus} \
        --tp_size ${tp_size} \
        --pp_size ${pp_size}
}

function build_engine_fp8_ifb {
    python ${folder_tekit}/examples/llama/build.py \
        --output_dir ${engine_dir} \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --enable_fp8 \
        --fp8_kv_cache \
        --strongly_typed \
        --enable_context_fmha \
        --remove_input_padding \
        --use_inflight_batching \
        --parallel_build \
        --n_positions ${n_positions} \
        --max_batch_size ${max_batch_size} \
        --max_input_len ${max_input_len} \
        --max_output_len ${max_output_len} \
        --max_num_tokens ${max_num_tokens} \
        --n_layer ${num_layers} \
        --n_head ${num_heads} \
        --n_kv_head ${num_kv_heads} \
        --n_embd ${hidden_size} \
        --inter_size ${inter_size} \
        --vocab_size ${vocab_size} \
        --world_size ${num_gpus} \
        --tp_size ${tp_size} \
        --pp_size ${pp_size}
}

function build_engine_int8sq_ifb {
    python ${folder_tekit}/examples/llama/build.py \
        --output_dir ${engine_dir} \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
	    --use_gemm_plugin float16 \
        --use_rmsnorm_plugin float16 \
        --int8_kv_cache \
        --use_smooth_quant \
        --enable_context_fmha \
        --remove_input_padding \
        --use_inflight_batching \
        --parallel_build \
        --n_positions ${n_positions} \
        --max_batch_size ${max_batch_size} \
        --n_layer ${num_layers} \
        --n_head ${num_heads} \
        --n_kv_head ${num_kv_heads} \
        --n_embd ${hidden_size} \
        --inter_size ${inter_size} \
        --vocab_size ${vocab_size} \
        --world_size ${num_gpus} \
        --tp_size ${tp_size} \
        --pp_size ${pp_size}
}

function benchmark_ifb {
    for output_len in 1 ${max_output_len}; do
        tag_benchmark=${tag_base}_bs${max_batch_size}_isl${max_input_len}_osl${output_len}_ifb
        mpirun --allow-run-as-root -n ${num_gpus} ${folder_tekit}/cpp/build/benchmarks/gptManagerBenchmark \
            --model ${model_base} \
            --engine_dir ${engine_dir} \
            --type IFB --dataset ${folder_tekit}/benchmarks/cpp/preprocessed_dataset.json \
            --max_num_sequences ${max_batch_size} \
            > ${folder_result}/cpp_${tag_benchmark}.txt
    done
}

folder_tekit=/home/l40s/eric/code/TensorRT-LLM
folder_engine=/home/l40s/eric/code/TensorRT-LLM/benchmarks/cpp/engine_tp
folder_result_base=./result_ifp
for folder in ${folder_engine} ${folder_result_base}; do
    if [[ ! -d ${folder} ]]; then mkdir ${folder}; fi
done

model_base=llama
batch_size_list="8 16 32 64 128"
input_len_output_len_list="1024,128,3000 2048,256,60000 3072,384,80000"
for precision in fp8; do
    for model_size in 70b; do
        set_param_${model_size}_tp1_pp2
        tag_base=${model_base}_${model_size}_${precision}_tp${tp_size}_pp${pp_size}
        folder_result=${folder_result_base}/${tag_base}
        if [[ ! -d ${folder_result} ]]; then mkdir ${folder_result}; fi
        for input_len_output_len in ${input_len_output_len_list}; do
            IFS=, read -r max_input_len max_output_len max_num_tokens <<< ${input_len_output_len}
            for max_batch_size in ${batch_size_list}; do
                tag_engine=${tag_base}_bs${max_batch_size}_isl${max_input_len}_osl${max_output_len}_ifb
                engine_dir=${folder_engine}/${tag_engine}
                path_config=${folder_result}/config_${tag_engine}.json
                if [[ ! -f ${path_config} ]]; then
                    build_engine_${precision}_ifb
                    benchmark_ifb
                    cp ${engine_dir}/config.json ${path_config}
                    rm -r ${engine_dir}
                fi
            done
        done
    done
done

