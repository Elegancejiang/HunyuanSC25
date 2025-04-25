import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import re
from pathlib import Path

def parse_exhaustive_data(content):
    """解析穷举数据"""
    exhaustive = {}
    # 按filename=分割不同的图
    graph_blocks = re.split(r'\n(?=graph:)', content)
    for block in graph_blocks:
        # 提取图名
        name_match = re.search(r'/([\w_-]+)\.graph', block)
        if not name_match: continue

        graph_name = name_match.group(1)
        print(graph_name)
        
        # 查找所有包含'edgecut='的行
        edgecuts = []
        for line in block.split('\n'):
            if 'edgecut=' in line:
                # 提取'edgecut='后面的数值
                value_match = re.search(r'edgecut=\s*(\d+)', line)
                if value_match:
                    edgecuts.append(int(value_match.group(1)))

        if edgecuts:
            exhaustive[graph_name] = edgecuts
    
    return exhaustive


def parse_metis_data(content):
    """解析METIS数据（提取前5个结果）"""
    metis = {}
    # 按filename分割不同的图
    graph_blocks = re.split(r'\n(?=filename=)', content)
    for block in graph_blocks:
        match = re.search(r'/([\w_-]+)\.graph', block)
        if not match: continue

        graph_name = match.group(1)
        edgecuts = []
        # 提取所有edgecut值
        for line in block.split('\n'):
            if 'edgecut=' in line:
                edgecut_match = re.search(r'edgecut=(\d+)', line)
                if edgecut_match:
                    edgecuts.append(int(edgecut_match.group(1)))
        # 保留前5个结果
        metis[graph_name] = edgecuts[:5]
    return metis

def parse_hunyuan_data(content):
    """解析HunyuanGraph数据（提取中间结果）"""
    hunyuan = {}
    # 按graph:分割不同的图
    graph_blocks = re.split(r'\n(?=graph:)', content)
    for block in graph_blocks:
        # 提取图名
        name_match = re.search(r'/([\w_-]+)\.graph', block)
        if not name_match: continue

        graph_name = name_match.group(1)
        
        # 查找所有包含'edgecut='的行
        edgecuts = []
        for line in block.split('\n'):
            if 'edgecut=' in line:
                # 提取'edgecut='后面的数值
                value_match = re.search(r'edgecut=\s*(\d+)', line)
                if value_match:
                    edgecuts.append(int(value_match.group(1)))

        if edgecuts:
            hunyuan[graph_name] = edgecuts
    
    return hunyuan

# 基础路径配置（假设脚本在项目根目录）
BASE_DIR = Path(__file__).parent.resolve()  # 获取当前脚本所在目录

# 定义相对路径组件
RELATIVE_PATHS = {
    "exhaustive_gpu": BASE_DIR / "init_partition/exhaustive_gpu/5090_exhaustive_1024_2_1.txt",
    "metis_cpu": BASE_DIR / "init_partition/metis_cpu/5090_metis_1024_2_0_metis.txt",
    "sampling_hunyuan": BASE_DIR / "init_partition/sampling_hunyuan/5090_sampling_1024_2_1.txt"
}

# 批量读取文件（推荐使用字典存储内容）
file_contents = {}
for file_type, path in RELATIVE_PATHS.items():
    try:
        with open(path, 'r', encoding='utf-8') as f:
            file_contents[file_type] = f.read()
    except FileNotFoundError:
        print(f"警告：文件 {file_type} 未找到，路径：{path}")

# 解包读取结果
doc1_content = file_contents["exhaustive_gpu"]
doc2_content = file_contents["metis_cpu"]
doc3_content = file_contents["sampling_hunyuan"]

# with open(r"D:\HuaweiMoveData\Users\huawei\Desktop\new\init partition\exhaustive_gpu\4090_exhaustive_1024_2_1.txt", 'r', encoding='utf-8') as f:
#     doc1_content = f.read()
# with open(r"D:\HuaweiMoveData\Users\huawei\Desktop\new\init partition\metis_cpu\4090_metis_1024_2_0_0404.txt", 'r', encoding='utf-8') as f:
#     doc2_content = f.read()
# with open(r"D:\HuaweiMoveData\Users\huawei\Desktop\new\init partition\sampling_hunyuan\4090_sampling_1024_2_1.txt", 'r', encoding='utf-8') as f:  # 新增数据文件
#     doc3_content = f.read()

exhaustive_data = parse_exhaustive_data(doc1_content)
metis_data = parse_metis_data(doc2_content)
hunyuan_data = parse_hunyuan_data(doc3_content)

# 指定的九个图
selected_graphs = [
    "wb-edu_gpu_1024",
    "amazon-2008_gpu_1024",
    "vas_stokes_4M_gpu_1024",
    "road_usa_gpu_1024",
    "nlpkkt120_gpu_1024",
    "Bump_2911_gpu_1024",
    "cage15_gpu_1024",
    "hugebubbles-00000_gpu_1024",
    "kron_g500-logn21_gpu_1024"
]
# selected_graphs = ['audikw_1_gpu_1024', 'af_shell1_gpu_1024', 'ML_Geer_gpu_1024', 'dielFilterV3clx_gpu_1024',
#                    'hugebubbles-00010_gpu_1024', 'Queen_4147_gpu_1024', 'road_usa_gpu_1024', 'rgg_n_2_22_s0_gpu_1024', 'Cube_Coup_dt6_gpu_1024']
# 交叉筛选出三个数据集中都存在的图
valid_graphs = [g for g in selected_graphs if g in exhaustive_data and g in metis_data and g in hunyuan_data]

def format_y_ticks(x, pos):
    """Format the y-axis ticks to display two decimal places"""
    return f'{x:.2f}' 

fig, axes = plt.subplots(3, 3, figsize=(35, 22))
axes = axes.flatten()

for i, graph_name in enumerate(valid_graphs):
    ax = axes[i]
    ex_values = exhaustive_data[graph_name]
    metis_values = metis_data[graph_name]
    hy_values = hunyuan_data[graph_name]
    
    # Debugging print
    print(f"Graph: {graph_name}")
    print(f"  Exhaustive values: {ex_values}")
    print(f"  METIS values: {metis_values}")
    print(f"  Hunyuan values: {hy_values}")
    print("-" * 50)
    
    # Skip if data is missing
    if not ex_values or not hy_values or not metis_values:
        print(f"Warning: Skipping {graph_name} due to missing data.")
        continue
    
    # Log transformation with zero handling
    ex_log = np.log10([x if x > 0 else 1e-10 for x in ex_values])
    metis_log = np.log10([x if x > 0 else 1e-10 for x in metis_values])
    hy_log = np.log10([x if x > 0 else 1e-10 for x in hy_values])
    
    x_pos = 1  # Shared x-coordinate for overlap
    
    # Exhaustive Violin Plot (plotted first = below)
    violin_ex = ax.violinplot(
        [ex_log],
        positions=[x_pos],
        widths=0.4,  # Narrower than 0.6
        showmedians=True
    )
    for pc in violin_ex['bodies']:
        pc.set_facecolor('#FFA500')  # Orange
        pc.set_edgecolor('#FFA500')
        pc.set_linewidth(2)
        pc.set_alpha(0.9)
    violin_ex['cmaxes'].set_color('#FFA500')
    violin_ex['cmins'].set_color('#FFA500')
    violin_ex['cbars'].set_color('#FFA500')
    violin_ex['cmedians'].set_color('black')
    
    # Hunyuan Violin Plot (plotted second = above)
    violin_hy = ax.violinplot(
        [hy_log],
        positions=[x_pos],
        widths=0.3,  # Narrower than Exhaustive
        showmedians=True
    )
    for pc in violin_hy['bodies']:
        pc.set_facecolor('#0804f9')  # Blue
        pc.set_edgecolor('#0804f9')
        pc.set_linewidth(2)
        pc.set_alpha(0.7)  # Show Exhaustive underneath
    violin_hy['cmaxes'].set_color('#0804f9')
    violin_hy['cmins'].set_color('#0804f9')
    violin_hy['cbars'].set_color('#0804f9')
    violin_hy['cmedians'].set_color('black')
    
    # METIS Scatter (plotted last, on top)
    ax.scatter(
        [x_pos] * len(metis_log),
        metis_log,
        color='red',
        edgecolor='black',
        s=400,
        marker='D',
        zorder=3
    )
    
    # Title and axes
    clean_title = graph_name.replace('_gpu_1024', '')
    ax.set_title(clean_title, fontsize=60)
    ax.tick_params(axis='y', labelsize=50)
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    ax.set_xticks([])  # Hide x-axis ticks
    
    # ax.set_yticks([min(ex_log + metis_log + hy_log), max(ex_log + metis_log + hy_log)])  # Set only two ticks at min and max
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks)) 
    
    if i % 3 == 0:
        ax.set_ylabel('Log(Cutsize)', fontsize=70)
        ax.yaxis.set_label_coords(-0.2, 0.5)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.7, 1.3)  # Center the plots
    
# Remove unused subplots
for j in range(len(valid_graphs), 9):
    fig.delaxes(axes[j])

# Global legend
legend_elements = [
    Patch(facecolor='#FFA500', edgecolor='#FFA500', label='Exhaustive'),
    Patch(facecolor='#0804f9', edgecolor='#0804f9', label='Hunyuan'),
    Line2D([0], [0],
           marker='D',
           color='w',
           markersize=40,
           markerfacecolor='red',
           markeredgecolor='black',
           label='METIS',
           linestyle='')
]
fig.subplots_adjust(top=0.88)
fig.legend(
    handles=legend_elements,
    loc='upper center',
    ncol=3,
    fontsize=60,
    framealpha=0.9,
    bbox_to_anchor=(0.5, 1.10),
    handlelength=1.5,
    handleheight=1.5,
    handletextpad=0.5,
    columnspacing=2
)

# Save with confirmation
plt.tight_layout()
plt.savefig("figure10.png", dpi=300, bbox_inches='tight')
print("Saved: figure10.png")
plt.savefig("figure10.pdf", bbox_inches='tight')
print("Saved: figure10.pdf")
plt.close()