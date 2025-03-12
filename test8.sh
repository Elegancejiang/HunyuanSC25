input="graph1.csv"

i=0 # 初始化计数器
while IFS=',' read -r Name; do
    j=0 # 初始化重复执行计数器
    while [ $j -lt 3 ]; do
        ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 8 0 >> test_8_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 16 0 >> test_16_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 32 0 >> test_32_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 64 0 >> test_64_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 128 0 >> test_128_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 256 0 >> test_256_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 512 0 >> test_512_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 1024 0 >> test_1024_0.txt

        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 8 1 >> test_8_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 16 1 >> test_16_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 32 1 >> test_32_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 64 1 >> test_64_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 128 1 >> test_128_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 256 1 >> test_256_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 512 1 >> test_512_1.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 1024 1 >> test_1024_1.txt
        j=$((j + 1))
    done
    i=$((i + 1)) # 更新计数器
done < "$input"

echo "Processed $i files."