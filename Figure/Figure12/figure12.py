import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker  # 导入 ticker 模块
import textwrap  # 导入 textwrap 模块
from matplotlib.colors import to_rgb
import os
from pathlib import Path

# 1. 读取数据文件
# 设置基础路径（相对路径的根目录）
BASE_DIR = Path(__file__).parent.resolve()  # 获取当前脚本所在目录

tmp_paths = {
    "Mt-Metis": os.path.join("uncoarsen", "mt_normalized_new.csv"),
    "ParMetis": os.path.join("uncoarsen", "parmetis_normalized_new.csv"),
    "Jet": os.path.join("uncoarsen", "jet_normalized_new.csv"),
    "Hunyuan": os.path.join("uncoarsen", "hunyuan_normalized_new.csv"),
    "KaMinPar": os.path.join("uncoarsen", "kaminpar_normalized_new.csv")
}

# 动态拼接完整路径（推荐）
file_paths = {k: str(BASE_DIR / v) for k, v in tmp_paths.items()}

# # 1. 读取数据文件
# file_paths = {
#     "Mt-Metis": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\uncoarsen\mt_normalized_new.csv",
#     "ParMetis": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\uncoarsen\parmetis_normalized_new.csv",
#     "Jet": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\uncoarsen\jet_normalized_new.csv",
#     "Hunyuan": r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\uncoarsen\hunyuan_normalized_new.csv",
#     # "KaHyPar":r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\coarsen\kahypar_ormalized.csv",
#     "KaMinPar":r"D:\HuaweiMoveData\Users\huawei\Desktop\three_phase\uncoarsen\kaminpar_normalized_new.csv"
# }
dataframes = {name: pd.read_csv(path, index_col=0) for name, path in file_paths.items()}

# 2. 设定固定的分区标签
partition_labels = ["8-part", "32-part", "128-part", "512-part"]

# 3. 计算所有数据集中共有的图名称
common_graphs = set.intersection(*(set(df.index) for df in dataframes.values()))
common_graphs = list(common_graphs)

# 4. 生成 Metis 基准数据 (始终为 1)
metis_df = pd.DataFrame(1, index=common_graphs, columns=partition_labels)
dataframes["Metis"] = metis_df

# 5. 确保所有数据集具有相同的列，并填充缺失值为 0
filtered_data = {
    name: df.reindex(common_graphs).fillna(0).reindex(columns=partition_labels, fill_value=0).astype(float)
    for name, df in dataframes.items()
}

# 6. 选择需要的 9 个图 (数据来源)
selected_graphs = [
    "wb-edu", "amazon-2008", "vas_stokes_4M", "road_usa","nlpkkt120",
    "Bump_2911",  "cage15", "hugebubbles-00000","kron_g500-logn21"
]

# 根据选择的图数据过滤出相应的子集
filtered_data_selected = {name: df.loc[selected_graphs] for name, df in filtered_data.items()}

# 设定绘制顺序
ordered_labels = ["Metis", "Mt-Metis", "ParMetis", "KaMinPar", "Jet", "Hunyuan"]

# 定义新的子图标题
new_graph_titles = [
    "wb-edu", "amazon-2008", "vas_stokes_4M", "road_usa","nlpkkt120",
    "Bump_2911",  "cage15", "hugebubbles-00000","kron_g500-logn21"
]
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
    '#6495ED', '#FFA500', '#0804f9'
]

# 提亮后的颜色
custom_colors = [brighten(c, 1.05) for c in custom_colors_base]

# 每张图显示 9 个子图，3x3 格式
fig, axes = plt.subplots(3, 3, figsize=(42, 33), sharey=False)
axes = axes.flatten()

# 设置柱子的宽度，减小柱子之间的间距
width = 0.15

for i, (graph, title) in enumerate(zip(selected_graphs, new_graph_titles)):
    ax = axes[i]

    x = np.arange(len(partition_labels))

    # Compute dynamic y-axis range
    y_max = max(df.loc[graph, partition_labels].max() for df in filtered_data_selected.values()) * 1.1
    y_min = 0

    # Plot bars with black edge outline
    for j, name in enumerate(ordered_labels):
        ax.bar(x + j * width, filtered_data_selected[name].loc[graph, partition_labels].values,
               width, label=name, color=custom_colors[j], edgecolor="black", linewidth=1.2)

    # Set x-axis labels
    wrapped_labels = [textwrap.fill(p.replace("-part", ""), width=5) for p in partition_labels]
    ax.set_xticks(x + width * len(ordered_labels) / 2)
    ax.set_xticklabels(wrapped_labels, fontsize=70)
    ax.set_title(title, fontsize=80)  # 使用新的标题
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='y', labelsize=70)

    # 使用 MaxNLocator 调整纵坐标刻度的数量
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))

    # Draw a black border around each subplot
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             linewidth=1.5, edgecolor='black', facecolor='none')
    
    ax.add_patch(rect)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.9)
    if i % 3 == 0:
        ax.set_ylabel("Speedup", fontsize=80, labelpad=30)

# Global legend at the top
wrapped_labels_legend = [textwrap.fill(label, width=12) for label in ordered_labels]  # 将legend标签换行
handles = [plt.Rectangle((0, 0), 1, 1, color=custom_colors[i], edgecolor="black", linewidth=1.2) 
           for i in range(len(ordered_labels))]
fig.legend(handles, wrapped_labels_legend, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=80, columnspacing=0.85)
# Adjust layout to improve spacing
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.8, wspace=0.25, hspace=0.35)

# Add Speedup label closer to the plots
# fig.text(0.02, 0.5, "Speedup", va='center', rotation='vertical', fontsize=80, fontweight='bold')

# Add X-axis label at the bottom
# fig.text(0.55, 0.08, "Number of Partitions", ha='center', fontsize=80, fontweight='bold')
for i in range(6, 9):  # Last row
    axes[i].set_xlabel("Number of Partitions", fontsize=70, labelpad=30)
# Save the figure with a unique name
output_filename = "figure12.png"
plt.savefig(output_filename, dpi=400, bbox_inches='tight')
plt.savefig("figure12.pdf", bbox_inches='tight')
plt.close()  # Close the figure to free up memory

# print(f"Saved the plot as {output_filename}")
