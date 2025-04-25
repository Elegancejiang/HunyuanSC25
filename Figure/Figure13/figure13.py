import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.resolve()  # 假设脚本位于 class 目录
RELATIVE_ROOT = BASE_DIR / "edge_time/geomean"

# 定义数据集类别（动态生成文件名）
categories = [
    "Web crawl",
    "Social network",
    "Semiconductor",
    "Road network",
    "Optimization",
    "Finite element",
    "Biology",
    "Artificial mesh",
    "Artificial complex"
]

# 动态生成文件路径列表
files = [RELATIVE_ROOT / f"Geometric_Mean_{cat}_8.csv" for cat in categories]

# # Load the CSV files
# files = [
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Web crawl_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Social network_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Semiconductor_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Road network_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Optimization_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Finite element_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Biology_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Artificial mesh_8.csv",
#     r"D:\HuaweiMoveData\Users\huawei\Desktop\class\edge_time\geomean\Geometric_Mean_Artificial complex_8.csv",
# ]

# dataframes = {file.split('\\')[-1].split('.')[0]: pd.read_csv(file) for file in files}  # Remove .csv from keys
dataframes = {file.stem: pd.read_csv(file) for file in files}

category_names = list(dataframes.keys())
geomean_values = []
labels = []

for category in category_names:
    df = dataframes[category]
    geomean_row = df[df['Graph Name'] == 'Geomean']
    if not geomean_row.empty:
        geomean_values.append(geomean_row.iloc[0, 1:].values)
        if not labels:
            labels = geomean_row.columns[1:].tolist()
    else:
        print(f"Warning: 'Geomean' row not found in {category}")

geomean_values = np.array(geomean_values)

def brighten(color, factor=1.3):
    """提高颜色亮度，factor > 1 表示提亮"""
    r, g, b = to_rgb(color)
    r = min(r * factor, 1.0)
    g = min(g * factor, 1.0)
    b = min(b * factor, 1.0)
    return (r, g, b)

# 原始颜色
custom_colors_base = [
    '#934B43', '#e68b8b', '#FFD700',
    '#3CB371', '#05B9E2', '#6495ED', '#FFA500', '#0804f9'
]

# 提亮后的颜色
custom_colors = [brighten(c, 1.05) for c in custom_colors_base]

legend_labels = ['METIS', 'Mt-Metis', 'ParMETIS', 'PT-Scotch', 'mt-KaHIP', 'KaMinPar', 'Jet', 'Hunyuan']

if len(labels) != len(legend_labels):
    raise ValueError(f"❌ 标签数量不一致！labels: {len(labels)} vs legend_labels: {len(legend_labels)}")

width = 0.3
x_shift = 0.3  # Reduced from 0.8 to minimize edge padding
category_gap = 2.0  # Increased from 1.5 to separate categories
x = np.arange(len(category_names))

fig, ax = plt.subplots(figsize=(42, 25))

for i, label in enumerate(labels):
    for j in range(len(category_names)):
        ax.bar(
            x[j] + i * width + x_shift + j * category_gap,
            geomean_values[j, i],
            width,
            color=custom_colors[i],
            edgecolor='black'
        )

# 坐标轴设置
ax.set_ylabel('Geomean Edge/Time (s)', fontsize=100)
# 提取文件名中的类别名称，确保移除 'Geometric_Mean_' 和 '_8'
xticks_labels = [
    category.replace('Geometric_Mean_', '').replace('_8', '').replace('.csv', '')
    for category in category_names
]
tick_positions = x + width * (len(labels) - 1) / 2 + x_shift + np.arange(len(category_names)) * category_gap
ax.set_xticks(tick_positions)
ax.set_xticklabels([name.title() for name in xticks_labels], fontsize=100, rotation=30, ha="right")
ax.tick_params(axis='y', labelsize=90)
ax.yaxis.get_offset_text().set_fontsize(70)

# 设置 x 轴范围，减少前后空白
# ax.set_xlim( x_shift, tick_positions[-1] + width * len(labels) )

# 网格与参考线
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
ax.xaxis.grid(False)

# 手动创建图例对象
legend_handles = [Patch(facecolor=custom_colors[i], edgecolor='black', label=legend_labels[i]) for i in range(len(legend_labels))]

# 图例放在顶部
fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.18),
    ncol=4,
    fontsize=90,
    title_fontsize=14,
    frameon=False,
    handletextpad=0.5,
    columnspacing=0.8,
    handlelength=1.2
)

plt.tight_layout()
output_filename = "figure13.png"
plt.savefig(output_filename, dpi=400, bbox_inches='tight')
plt.savefig("figure13.pdf", bbox_inches='tight')