# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt
# ./hunyuangraph /media/jiangdie/shm_ssd/graph_10w/vas_stokes_1M.graph 2 >> test.txt

make
# ./hunyuangraph graph.txt 2 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/vas_stokes_1M_gpu.graph 2 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 8 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2 > test.txt
# ./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/rgg_n_2_22_s0_gpu.graph 2 > test.txt

./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 2 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 8 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 32 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 64 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 128 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 256 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/audikw_1_gpu.graph 512 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 1024 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 2048 >> test.txt
./hunyuangraph /home/jiangdie/study/mygp_0.9.9_test_initpartition/graph/hugebubbles-00000_gpu.graph 4096 >> test.txt

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