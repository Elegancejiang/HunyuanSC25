#!/bin/bash

input="graph1.csv"
# input="graph_sampling.csv"
# p_values="8 16 32 64 128 256 512 1024"  # 改为字符串，用空格分隔
p_values="8"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        j=0
        while [ $j -lt 1 ]; do
            ./hunyuangraph /media/jiangdie/新加卷/graph_10w/${Name}.graph $p 1 >> syncfree.txt
            # ./hunyuangraph /home/lyj/HunyuanGraph_SC25_0.0.0_03141426/graph/${Name}.graph 2 0 >> 4090_sampling_2_0.txt
            # mv graph.txt ${Name}_gpu_8.graph
            # mv ${Name}_gpu_8.graph /home/lyj/HunyuanGraph_SC25_0.0.0_03141426/graph
            j=$((j + 1))
        done
        echo "Processed $i files."
        i=$((i + 1))
    done < "$input"
    echo "Processed $p partitions."
done