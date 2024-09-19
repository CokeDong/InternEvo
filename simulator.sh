#!/bin/bash
set +x
# 定义数组，包含所有的配置文件路径
configs=(
    "configs/70B_llama2.py" # standard llama2 70B
    "configs/7B_isp_sft.py" # 7B 32k seqlen
    "configs/72B_qwen2.py"   # 72B qwen
)

# 遍历数组中的每个配置文件
for config in "${configs[@]}"; do
    configname="${config##*/}"
    echo $configname
    # 构建 srun 命令并执行
    srun -p Intern5 python simulation_train_formulaic.py --pre_profiling_data_path ./prof_data/data.pt --config "$config" --run_all_solu --world_size 512 --global_batch_size 4194304 >> ./simulator_$configname.log 2>&1
done

