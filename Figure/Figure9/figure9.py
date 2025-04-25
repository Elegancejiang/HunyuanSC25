import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path

# ========== 提取 Hunyuan 边切值数据 ==========
def extract_hunyuan_edgecut(file_path):
    result = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_graph = ""
    values = []
    for line in lines:
        graph_match = re.search(r'graph:.*/(.*?)\.graph', line)
        level_match = re.search(r'level\s+(\d+):.*adjwgtsum\s+(\d+)', line)
        if graph_match:
            if current_graph != "":
                result[current_graph] = values
                values = []
            current_graph = graph_match.group(1)
        elif level_match:
            values.append(int(level_match.group(2)))

    if current_graph != "":
        result[current_graph] = values
    return result

# ========== 提取 Jet 边切值数据 ==========
def extract_jet_edgecut(file_path):
    result = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_graph = ""
    values = []
    for line in lines:
        graph_match = re.search(r'Reading.*?/([^/]+?)\.graph', line)
        level_match = re.search(r'level=(\d+)\s+nvtxs=.*?adjwgt_sum=(\d+)', line)
        if graph_match:
            if current_graph != "":
                result[current_graph] = values
                values = []
            current_graph = graph_match.group(1)
        elif level_match:
            values.append(int(level_match.group(2)))

    if current_graph != "":
        result[current_graph] = values
    return result

# ========== 提取 Hunyuan 时间数据 ==========
def extract_hunyuan_time(file_path):
    result = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_graph = ""
    times = []
    for line in lines:
        graph_match = re.search(r'graph:\s*(?:.*[/\\])?([^/\\]+)\.graph', line, re.IGNORECASE)
        if graph_match:
            if current_graph and times:
                result[current_graph] = times
                times = []
            current_graph = graph_match.group(1).strip()
        time_match = re.search(r'level=\s*\d+\s+time=\s*([\d.]+)\s*ms', line)
        if time_match:
            times.append(float(time_match.group(1)))

    if current_graph and times:
        result[current_graph] = times
    return result

# ========== 提取 Jet 时间数据 ==========
def extract_jet_time(file_path):
    result = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_graph = ""
    times = []
    for line in lines:
        graph_match = re.search(r'Reading.*?/([^/]+?)\.graph', line)
        time_match = re.search(r'level=\s+(\d+)\s+time=\s+([\d.]+)', line)
        if graph_match:
            if current_graph and times:
                result[current_graph] = times
                times = []
            current_graph = graph_match.group(1)
        elif time_match:
            times.append(float(time_match.group(2)))

    if current_graph and times:
        result[current_graph] = times
    return result

def plot_data(final_data):
    plt.rcParams.update(plt.rcParamsDefault)
    COLOR_PALETTE = {
        'jet': {'base': '#FFA500', 'gradient': ['#7EB5DA', '#0A3D6B'], 'marker': 'D'},
        'hunyuan': {'base': '#0804f9', 'gradient': ['#FFB300', '#C43E00'], 'marker': 's'}
    }
    BENCHMARK_GRAPHS = [
        "wb-edu", "amazon-2008", "vas_stokes_4M", "road_usa", "nlpkkt120",
        "Bump_2911", "cage15", "hugebubbles-00000", "kron_g500-logn21"
    ]
    valid_graphs = [g for g in BENCHMARK_GRAPHS if g in final_data]

    fig = plt.figure(figsize=(40, 26), facecolor='white')
    gs = GridSpec(3, 3, figure=fig, left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.4, hspace=0.55)

    def _plot_algo(ax, times, edges, style):
        ax.plot(times, edges, color=style['base'], lw=4.5, marker=style['marker'],
                markersize=8, markevery=1, markerfacecolor=style['base'],
                markeredgecolor=style['base'], markeredgewidth=3, zorder=20)

    for idx, graph in enumerate(valid_graphs):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        data = final_data[graph]

        jet_t = np.array(data["jet"]["time"])
        jet_e = np.array(data["jet"]["edgecut"])
        hy_t = np.array(data["hunyuan"]["time"])
        hy_e = np.array(data["hunyuan"]["edgecut"])

        ax.tick_params(axis='both', which='major', length=12, width=3, pad=12,
                       bottom=True, top=False, left=True, right=False)

        min_len = min(len(jet_t), len(jet_e), len(hy_t), len(hy_e))
        if min_len == 0:
            ax.set_visible(False)
            continue

        jet_t = np.insert(jet_t[:min_len], 0, 0)
        jet_e = np.insert(jet_e[:min_len], 0, jet_e[0])
        hy_t = np.insert(hy_t[:min_len], 0, 0)
        hy_e = np.insert(hy_e[:min_len], 0, hy_e[0])

        _plot_algo(ax, jet_t, jet_e, COLOR_PALETTE['jet'])
        _plot_algo(ax, hy_t, hy_e, COLOR_PALETTE['hunyuan'])

        ax.set_xlim(left=0)
        is_last_row = row == 2
        is_first_col = col == 0

        ax.set_xlabel('Time (ms)' if is_last_row else '', fontsize=70, labelpad=18)
        ax.set_ylabel('Edge Weight Sum' if is_first_col else '', fontsize=61, labelpad=22)
        ax.tick_params(axis='y', labelsize=60)
        ax.tick_params(axis='x', labelsize=60)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.yaxis.set_major_formatter(ticker.EngFormatter(places=0))
        ax.tick_params(axis='both', which='major', length=12, width=3, pad=12,
                       colors='000000', bottom=True, top=False, left=True, right=False)

        ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.6, color='grey')
        ax.set_title(graph, fontsize=70, pad=10, fontweight='normal')

        for spine in ax.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(5)

    legend_elements = [
        Line2D([], [], color=COLOR_PALETTE['jet']['base'], marker='D',
               markersize=18, lw=4, label='Jet'),
        Line2D([], [], color=COLOR_PALETTE['hunyuan']['base'], marker='s',
               markersize=18, lw=4, label='Hunyuan')
    ]

    fig.legend(handles=legend_elements, loc='upper center', edgecolor='black',
               bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True, framealpha=0.9,
               borderpad=0.8, handletextpad=0.8, columnspacing=1.5, labelspacing=0.5,
               fontsize=70)

    # os.makedirs('plots', exist_ok=True)
    plt.savefig('figure9.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure9.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ========== 主程序 ==========
def main():
    # 基础路径设置（推荐使用pathlib）
    BASE_DIR = Path(__file__).parent.resolve()  # 获取当前脚本所在目录[3,5](@ref)
    RELATIVE_ROOT = BASE_DIR                    # 公共子目录

    edgecut_files = {
        "hunyuan": RELATIVE_ROOT / "5090_hunyuan_1_8_coarsen_adjwgtsum.txt",
        "jet": RELATIVE_ROOT / "5090_jet_8_coarsen_adjwgtsum.txt"
    }

    time_files = {
        "hunyuan": RELATIVE_ROOT / "5090_hunyuan_1_8_coarsen_time.txt",
        "jet": RELATIVE_ROOT / "5090_jet_8_coarsen_time.txt"
    }

    # edgecut_files = {
    #     "hunyuan": r"D:\Download\wechat download\WeChat Files\wxid_61djvxkv4osh32\FileStorage\File\2025-04\5090_hunyuan_1_8_coarsen_adjwgtsum.txt",
    #     "jet": r"D:\Download\wechat download\WeChat Files\wxid_61djvxkv4osh32\FileStorage\File\2025-04\5090_jet_8_coarsen_adjwgtsum.txt"
    # }
    # time_files = {
    #     "hunyuan": r"D:\Download\wechat download\WeChat Files\wxid_61djvxkv4osh32\FileStorage\File\2025-04\5090_hunyuan_1_8_coarsen_time.txt",
    #     "jet": r"D:\Download\wechat download\WeChat Files\wxid_61djvxkv4osh32\FileStorage\File\2025-04\5090_jet_8_coarsen_time.txt"
    # }

    # 提取数据
    print("=== Extracting Hunyuan Edgecut ===")
    hunyuan_edgecut = extract_hunyuan_edgecut(edgecut_files["hunyuan"])
    for graph, values in hunyuan_edgecut.items():
        print(f"Graph: {graph}, Edgecut Values: {values}")

    print("\n=== Extracting Jet Edgecut ===")
    jet_edgecut = extract_jet_edgecut(edgecut_files["jet"])
    for graph, values in jet_edgecut.items():
        print(f"Graph: {graph}, Edgecut Values: {values}")

    print("\n=== Extracting Hunyuan Time ===")
    hunyuan_time = extract_hunyuan_time(time_files["hunyuan"])
    for graph, times in hunyuan_time.items():
        print(f"Graph: {graph}, Times (ms): {times}")

    print("\n=== Extracting Jet Time ===")
    jet_time = extract_jet_time(time_files["jet"])
    for graph, times in jet_time.items():
        print(f"Graph: {graph}, Times (ms): {times}")

    # 组合最终数据
    print("\n=== Final Combined Data ===")
    final_data = {graph: {
        "jet": {"edgecut": jet_edgecut.get(graph, []), "time": jet_time.get(graph, [])},
        "hunyuan": {"edgecut": hunyuan_edgecut.get(graph, []), "time": hunyuan_time.get(graph, [])}
    } for graph in set(hunyuan_edgecut.keys()).union(jet_edgecut.keys())}
    for graph, data in final_data.items():
        print(f"Graph: {graph}")
        print(f"  Jet Edgecut: {data['jet']['edgecut']}")
        print(f"  Jet Time: {data['jet']['time']}")
        print(f"  Hunyuan Edgecut: {data['hunyuan']['edgecut']}")
        print(f"  Hunyuan Time: {data['hunyuan']['time']}")

    # 保存 JSON
    with open('coarsening_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
    print("\nFinal data saved to coarsening_data.json")

    # 绘图
    plot_data(final_data)
    # print("Plots saved to plots/benchmark_fullsize.pdf and plots/benchmark_fullsize.png")

if __name__ == "__main__":
    main()