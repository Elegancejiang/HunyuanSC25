
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_2M_gpu.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/audikw_1.graph 8 >> coarsen_SC25.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/rgg_n_2_22_s0.graph 8 >> coarsen_SC25.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/hugebubbles-00000.graph 8 >> coarsen_SC25.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/vas_stokes_2M.graph 8 >> coarsen_SC25.txt

input="graph1.csv"

i=0 # 初始化计数器
while IFS=',' read -r Name; do
    # ./hunyuangraph_SC24 /media/jiangdie/新加卷/graph_10w/${Name}.graph 8 1 >> random_match_SC24.txt
    # ./hunyuangraph /media/jiangdie/新加卷/graph_10w/${Name}.graph 8 1 >> random_match_syncfree.txt
    # ./hunyuangraph_randommatch_syncfree /media/jiangdie/新加卷/graph_10w/${Name}.graph 8 1 >> random_match_syncfree_exam.txt
    ./hunyuangraph /media/jiangdie/新加卷/graph_10w/${Name}.graph 8 1 >> test.txt
    echo "Processed $i files."
    i=$((i + 1)) # 更新计数器
done < "$input"

# make
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_2M_gpu.graph 2 >> test.txt

# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 8 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 8 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 8 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_2M_gpu.graph 8 >> test.txt

# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 64 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 64 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 64 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_2M_gpu.graph 64 >> test.txt

# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 256 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 256 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 256 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_2M_gpu.graph 256 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 64 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 256 >> test.txt

# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/audikw_1.graph 64 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/rgg_n_2_22_s0.graph 64 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/hugebubbles-00000.graph 64 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/vas_stokes_2M.graph 64 >> test.txt

# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/audikw_1.graph 256 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/rgg_n_2_22_s0.graph 256 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/hugebubbles-00000.graph 256 >> test.txt
# ./hunyuangraph /media/jiangdie/新加卷/graph_10w/vas_stokes_2M.graph 256 >> test.txt
# ./hunyuangraph graph.txt 2 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_1M_gpu.graph 2 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 8 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 2 > test.txt

# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 2 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 4 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 8 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 32 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 64 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 128 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 256 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 512 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 1024 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2048 >> test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 4096 >> test.txt

# 循环10次执行命令
# i=1
# while [ $i -le 10 ]
# do
#     echo "Running iteration $i"
#     ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2 >> test.txt
#     i=$((i + 1))
#     done
# echo "All iterations completed."

# python3 init.py > 1.txt
# python3 exammemory.py > 1.txt