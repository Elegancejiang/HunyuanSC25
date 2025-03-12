input="graph1.csv"

i=0 # 初始化计数器
while IFS=',' read -r Name; do
    j=0 # 初始化重复执行计数器
    while [ $j -lt 3 ]; do
        ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 512 0 >> test_512_0.txt
        # ./hunyuangraph /home/yongjiang/graph_10w/${Name}.graph 512 1 >> test_512_1.txt
        j=$((j + 1))
    done
    i=$((i + 1)) # 更新计数器
done < "$input"

echo "Processed $i files."