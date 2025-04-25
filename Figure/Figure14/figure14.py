import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.resolve()  # 假设脚本位于 class 目录
RELATIVE_ROOT = BASE_DIR / "class"

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
files = [RELATIVE_ROOT / f"{cat}/cut_geomean.csv" for cat in categories]


# # CSV 文件路径列表
# files = [
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Web crawl\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Social network\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Semiconductor\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Road network\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Optimization\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Finite element\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Biology\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Artificial mesh\cut_geomean.csv',
#     r'C:\Users\liyon\Desktop\信息研23_2\SC25\graph\Figure14\class\Artificial complex\cut_geomean.csv',
#     # r"D:\HuaweiMoveData\Users\huawei\Desktop\class\Geomean\Geometric_Mean_cut.csv"
# ]

# dataframes = {file.split('\\')[-2]: pd.read_csv(file) for file in files}
dataframes = {file.parent.name: pd.read_csv(file)for file in files}
# category_names = list(dataframes.keys())
category_names = categories
print("category_names:", category_names)

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

def brighten(color, factor=1.0):
    """提高颜色亮度，factor > 1 表示提亮"""
    r, g, b = to_rgb(color)
    r = min(r * factor, 1.0)
    g = min(g * factor, 1.0)
    b = min(b * factor, 1.0)
    return (r, g, b)

# 原始颜色
custom_colors_base = [
    '#934B43', '#e68b8b', '#FFD700',
    '#3CB371', '#05B9E2', '#6495ED','#FFA500'
]

# 提亮后的颜色
custom_colors = [brighten(c, 1.05) for c in custom_colors_base]

legend_labels = ['METIS', 'Mt-Metis', 'ParMETIS', 'PT-Scotch', 'mt-KaHIP', 'KaMinPar', 'Jet']

if len(labels) != len(legend_labels):
    raise ValueError(f"❌ 标签数量不一致！labels: {len(labels)} vs legend_labels: {len(legend_labels)}")

width = 0.25
x_shift = 0.6
x = np.arange(len(category_names))

fig, ax = plt.subplots(figsize=(43, 25))

# 绘制条形图并添加数值标注
for i, label in enumerate(labels):
    bar_container = ax.bar(
        x + i * width + x_shift + np.arange(len(category_names)) * 1.5,
        geomean_values[:, i],
        width,
        color=custom_colors[i],
        edgecolor='black'
    )

    # 为每个柱子添加数值标注，只对高度大于 2 的柱子进行标注
    for bar in bar_container:
        height = bar.get_height()
        if height > 3:  # 只有当高度大于 2 时，才显示数值
            ax.text(
                bar.get_x() + bar.get_width() / 4, height -1.95, f'{height:.1f}',
                ha='center', va='bottom', fontsize=80, color='black'
            )
 # y=1 在 (-2 + top_value)/2，接近中间
yticks = [0, 0.5, 1]
# 坐标轴设置
ax.set_ylabel('Geomean Cutsize Normalized to Ours', fontsize=93, y=0.55)
ax.set_xticks(x + width * (len(labels) - 1) / 2 + x_shift + np.arange(len(category_names)) * 1.5)
ax.set_xticklabels([name.title() for name in category_names], fontsize=100, rotation=30, ha="right")
ax.tick_params(axis='y', labelsize=90)

# 网格与参考线
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
ax.axhline(y=1, color='grey', linestyle='--', alpha=0.7, linewidth=2)
ax.xaxis.grid(False)
ax.set_ylim(0, 2.1)

# 手动创建图例对象（确保颜色对齐）
legend_handles = [Patch(facecolor=custom_colors[i], edgecolor='black', label=legend_labels[i]) for i in range(len(legend_labels))]

# 图例设置
fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.21),  # 图例位置稍微向上移动
    ncol=4,  # 设置为4列
    fontsize=90,
    title_fontsize=14,
    frameon=False,
    handletextpad=0.5,
    columnspacing=0.8,
    handlelength=1.2
)

plt.tight_layout()
plt.savefig("figure14.png", dpi=400, bbox_inches='tight')
plt.savefig("figure14.pdf", bbox_inches='tight')
# plt.show()
