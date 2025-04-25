import pandas as pd
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')

# 定义文件名列表
file_names = [
    '5090_hunyuan_1_8_9.csv',
    '5090_hunyuan_1_32_9.csv',
    '5090_hunyuan_1_128_9.csv',
    '5090_hunyuan_1_512_9.csv'
]

# 构建完整文件路径
files = [os.path.join(data_dir, fname) for fname in file_names]

# 读取所有CSV文件到DataFrame列表
data_frames = [pd.read_csv(file) for file in files]

# 定义固定的排序顺序
custom_order = [
    "wb-edu", "amazon-2008", "vas_stokes_4M", "road_usa", "nlpkkt120",
    "Bump_2911", "cage15", "hugebubbles-00000", "kron_g500-logn21"
]

# 提取数据并添加自定义排序逻辑
partition_sizes = [8, 32, 128, 512]
merged_data = []

for df, size in zip(data_frames, partition_sizes):
    # 确保生成正确的分区标签
    partition_label = f"{size}-part"
    
    # 转换Graph Name为分类数据类型
    df['Graph Name'] = pd.Categorical(
        df['Graph Name'], 
        categories=custom_order,
        ordered=True
    )
    # 按自定义顺序排序
    df_sorted = df.sort_values('Graph Name')
    
    temp_df = df_sorted[['Graph Name', 'Coarsen Time']].copy()
    temp_df['Partition Size'] = partition_label  # 使用正确的标签
    merged_data.append(temp_df)

# 合并所有DataFrame
final_df = pd.concat(merged_data, ignore_index=True)

# 数据透视（关键修复：确保生成所有预期的列）
pivoted_df = final_df.pivot_table(
    index='Graph Name', 
    columns='Partition Size', 
    values='Coarsen Time', 
    aggfunc='last'
)

# 确保所有预期的列存在（新增保护逻辑）
expected_columns = ['8-part', '32-part', '128-part', '512-part']
for col in expected_columns:
    if col not in pivoted_df.columns:
        pivoted_df[col] = 0  # 添加缺失列并填充0

# 列排序
pivoted_df = pivoted_df[expected_columns]

# 最终按自定义顺序筛选
pivoted_df_filled = pivoted_df.fillna(0).reindex(custom_order)

# 创建保存目录
save_dir = os.path.join(current_dir, 'figure8')
os.makedirs(save_dir, exist_ok=True)

output_file = os.path.join(save_dir, 'hunyuan_normalized_new.csv')
pivoted_df_filled.to_csv(output_file)

print(f'CSV文件已保存至: {output_file}')