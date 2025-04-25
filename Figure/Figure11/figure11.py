import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgb
import os
from pathlib import Path

# 1. 读取数据文件
# 设置基础路径（相对路径的根目录）
BASE_DIR = Path(__file__).parent.resolve()  # 获取当前脚本所在目录

tmp_paths = {
    "Jet": os.path.join("init", "jet-edge.csv"),
    "Hunyuan": os.path.join("init", "hunyuan_new.csv"),
}

# 动态拼接完整路径（推荐）
file_paths = {k: str(BASE_DIR / v) for k, v in tmp_paths.items()}

# # 1. 读取数据文件
# file_paths = {
#     "Jet": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\init\jet-edge.csv",
#     "Hunyuan": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\init\hunyuan_new.csv",
#     # "KaMinPar": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\init\kaminpar_new.csv",
#     # "KaHyPar": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\init\kahypar_edge.csv"
# }

dataframes = {name: pd.read_csv(path, index_col=0) for name, path in file_paths.items()}

# 2. 设定固定的分区标签
partition_labels = ["8-part", "32-part", "128-part", "512-part"]

# 3. 计算所有数据集中共有的图名称
common_graphs = set.intersection(*(set(df.index) for df in dataframes.values()))
common_graphs = list(common_graphs)  # 转换为列表，方便排序和索引

# 去除 common_graphs 中的重复项
common_graphs = list(set(common_graphs))

# 4. 确保所有数据集具有相同的列，并填充缺失值为 0
filtered_data = {}
for name, df in dataframes.items():
    # 去除重复的行索引和列索引
    df = df.loc[~df.index.duplicated()]  # 删除重复行
    df = df.loc[:, ~df.columns.duplicated()]  # 删除重复列
    
    # 进行 reindex 操作，确保所有数据集具有相同的行和列
    filtered_data[name] = df.reindex(common_graphs).fillna(0).reindex(columns=partition_labels, fill_value=0).astype(float)

# 5. 选择需要的 9 个图
selected_graphs = [
    "wb-edu", "amazon-2008", "vas_stokes_4M", "road_usa","nlpkkt120",
    "Bump_2911",  "cage15", "hugebubbles-00000","kron_g500-logn21"
]

# 根据选择的图数据过滤出相应的子集
filtered_data_selected = {name: df.loc[selected_graphs] for name, df in filtered_data.items()}

# 6. 设定绘制顺序
ordered_labels = [ "Jet", "Hunyuan"]

def brighten(color, factor=1.3):
    """提高颜色亮度，factor > 1 表示提亮"""
    r, g, b = to_rgb(color)
    r = min(r * factor, 1.0)
    g = min(g * factor, 1.0)
    b = min(b * factor, 1.0)
    return (r, g, b)

# 原始颜色
custom_colors_base = [
    '#FFA500', '#0804f9'
]

# 提亮后的颜色
custom_colors = [brighten(c, 1.05) for c in custom_colors_base]

# 每张图显示 9 个子图，3x3 格式
fig, axes = plt.subplots(3, 3, figsize=(42, 33), sharey=False)  # 3x3 布局
axes = axes.flatten()

# 格式化函数：只格式化显示，实际数据不除
def format_million(x, pos):
    """格式化Y轴，显示为 X.YM"""
    return f'{x / 1e6:.1f}M'  # 显示为 X.YM

for i, graph in enumerate(selected_graphs):
    ax = axes[i]

    x = np.arange(len(partition_labels))
    width = 0.3

    # 保留原始数值，不除以 1e6
    y_max_raw = max(df.loc[graph, partition_labels].max() for df in filtered_data_selected.values())
    y_max_normalized = y_max_raw * 1.1  # 不除单位，padding

    # 使用原始数据绘图（不缩放）
    for j, name in enumerate(ordered_labels):
        values = filtered_data_selected[name].loc[graph, partition_labels].values
        ax.bar(x + j * width, values, width, label=name, color=custom_colors[j],
               edgecolor="black", linewidth=1.2)

    ax.set_xticks(x + width * len(ordered_labels) / 2)
    ax.set_xticklabels([p.replace("-part", "") for p in partition_labels], fontsize=70)
    ax.set_title(graph, fontsize=80, pad=10)
    ax.set_ylim(0, min(y_max_normalized, 10 * 1e6))  # 保持在百万单位
    ax.tick_params(axis='y', labelsize=65)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_million))  # 添加格式化器

    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.9)

    if i % 3 == 0:
        ax.set_ylabel("Edges/time (s)", fontsize=70, labelpad=30)


# Global legend at the top
handles = [plt.Rectangle((0, 0), 1, 1, color=custom_colors[i], edgecolor="black", linewidth=1.2)
           for i in range(len(ordered_labels))]
labels = ordered_labels

# Adjust ncol to 3 to display 3 items in each row
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, fontsize=80)

# Adjust layout to improve spacing and accommodate exponents
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.8, wspace=0.25, hspace=0.5)

# Add X-axis labels for each column
for i in range(6, 9):  # Last row
    axes[i].set_xlabel("Number of Partitions", fontsize=70, labelpad=30)

# Save the figure with a unique name
plt.savefig("figure11.png", dpi=400, bbox_inches='tight')
plt.savefig("figure11.pdf", bbox_inches='tight')
plt.close()  # Close the figure to free up memory
