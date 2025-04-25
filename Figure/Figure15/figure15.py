import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Data setup
graphs = ["wb-edu", "amazon-2008", "vas_stokes_4M", "road_usa","nlpkkt120",
    "Bump_2911",  "cage15", "hugebubbles-00000","kron_g500-logn21"]
software = ["Mt-Metis", "ParMetis", "KaMinPar", "Jet", "Hunyuan"]

# Raw timing data [Coarsening, Initial, Uncoarsening] in milliseconds
data = {
    "wb-edu": [
        [830.86,679.14,126.02],  # Mt-Metis
        [507,20478,183],  # ParMetis
        [633,506,252],      # KaMinPar
        [2328.3900000000003,1.34,82.10000000000001],    # Jet
        [85.2,1.674,72.828]      # Hunyuan
    ],
    "amazon-2008": [
        [24.18,233.73,59.8],     # Mt-Metis
        [86,87,39],    # ParMetis
        [140,373,81],       # KaMinPar
        [26.66,1.09,47.739999999999995],     # Jet
        [12.925,1.434,28.122]       # Hunyuan
    ],
    "vas_stokes_4M": [
        [487.71999999999997,539.2900000000001,205.08], # Mt-Metis
        [650,42,248],# ParMetis
        [951,195,248],     # KaMinPar
        [117.13,0.66,82.29],    # Jet
        [53.410000,1.818000,38.438000]       # Hunyuan
    ],
    "road_usa": [
        [353.28,495.09999999999997,177.96], # Mt-Metis
        [656,333,178],  # ParMetis
        [1541,160,378],     # KaMinPar
        [59.53,0.53,87.25],    # Jet
        [57.440000,1.625000,96.104000]     # Hunyuan
    ],
    "nlpkkt120": [
        [135.80999999999997,215.06,131.41],  # Mt-Metis
        [243,5,116],  # ParMetis
        [597,223,159],      # KaMinPar
        [51.45,0.71,62.87],    # Jet
        [36.405000,1.880000,36.036000]      # Hunyuan
    ],
    "Bump_2911": [
        [209.72,123.5,63.72],   # Mt-Metis
        [140,6,68],  # ParMetis
        [507,197,167],      # KaMinPar
        [146.17,0.76,63.94],   # Jet
        [26.557000,1.596000,26.060000]      # Hunyuan
    ],
    "cage15": [
        [419.97,2036.67,318.46999999999997],  # Mt-Metis
        [419,30,275], # ParMetis
        [946,194,207],      # KaMinPar
        [63.12,1.14,79.31],     # Jet
        [40.933000,1.411000,49.020000]    # Hunyuan
    ],
    "hugebubbles-00000": [
        [353.60999999999996,367.14000000000004,190.26000000000002], # Mt-Metis
        [708,6,195],  # ParMetis
        [1647,187,269],     # KaMinPar
        [41.79,0.48000000000000004,92.39],    # Jet
        [27.417000,1.752000,65.414000]      # Hunyuan
    ],
    "kron_g500-logn21": [
        [1887.06,30793.17,3047.9500000000003],    # Mt-Metis
        [15437,40079,778],     # ParMetis
        [1606,1370,648],        # KaMinPar
        [2769.11,1.1900000000000002,474.37],    # Jet
        [493.846000,1.050000,533.186000]       # Hunyuan
    ]
}

# Calculate total times and percentages per graph
total_times = {graph: [] for graph in graphs}
percentages = {graph: [] for graph in graphs}
for graph in graphs:
    for i in range(len(software)):
        c, i_val, u = data[graph][i]
        total = c + i_val + u
        total_times[graph].append(total)
        percentages[graph].append({
            'Coarsening': (c / total * 100) if total > 0 else 0,
            'Initial': (i_val / total * 100) if total > 0 else 0,
            'Uncoarsening': (u / total * 100) if total > 0 else 0
        })

# Normalize total times inversely relative to Hunyuan (index 4) for each graph
normalized_times = {graph: [] for graph in graphs}
for graph in graphs:
    hunyuan_total = total_times[graph][4]  # Hunyuan is at index 4
    if hunyuan_total == 0:  # Use a small placeholder to avoid division by zero
        hunyuan_total = 0.001
        normalized_times[graph] = [0 if t == 0 else 0.001 / t for t in total_times[graph]]  # Inverse for others
    else:
        normalized_times[graph] = [hunyuan_total / t if t > 0 else 0 for t in total_times[graph]]

# Normalize total times inversely relative to Jet (index 3) for each graph
# normalized_times = {graph: [] for graph in graphs}
# for graph in graphs:
#     jet_total = total_times[graph][3]  # Jet is at index 3
#     if jet_total == 0:  # Use a small placeholder to avoid division by zero
#         jet_total = 0.001
#         normalized_times[graph] = [0 if t == 0 else 0.001 / t for t in total_times[graph]]  # Inverse for others
#     else:
#         normalized_times[graph] = [jet_total / t if t > 0 else 0 for t in total_times[graph]]
# Base colors (slightly deeper soft tones)
base_colors = [ '#e68b8b', '#FFD700'  , '#6495ED', '#FFA500', '#0804f9']

# Create gradient colormaps (light to dark: top to bottom)
gradients = {}
for i, base_color in enumerate(base_colors):
    gradients[software[i]] = LinearSegmentedColormap.from_list(
        f"{software[i]}_gradient", 
        [f"{base_color}40", f"{base_color}A0", f"{base_color}FF"],  # Light, Medium, Deeper
        N=3
    )

# Set up the figure
plt.figure(figsize=(42, 23))

# Positions for 45 bars with gaps between graph groups
bar_width = 1
group_gap = 1.0
x_positions = []
for i in range(9):
    start = i * (5 + group_gap)
    x_positions.extend(np.arange(start, start + 5))

# Prepare data for plotting
coarsen = []
init = []
uncoarsen = []
colors = []
for graph in graphs:
    for i, sw in enumerate(software):
        norm_total = normalized_times[graph][i]
        perc = percentages[graph][i]
        coarsen.append(norm_total * (perc['Coarsening'] / 100))
        init.append(norm_total * (perc['Initial'] / 100))
        uncoarsen.append(norm_total * (perc['Uncoarsening'] / 100))
        cmap = gradients[sw]
        colors.append([cmap(2), cmap(1), cmap(0)])  # Deeper (Uncoarsening), Medium (Init), Light (Coarsening)

# Plot bars with black outlines
for i in range(45):
    if coarsen[i] == 0 and init[i] == 0 and uncoarsen[i] == 0:
        coarsen[i] = init[i] = uncoarsen[i] = 0.0001  # Avoid zero-height bars
    plt.bar(x_positions[i], uncoarsen[i], color=colors[i][0], width=bar_width, edgecolor='black', linewidth=0.7)
    plt.bar(x_positions[i], init[i], bottom=uncoarsen[i], color=colors[i][1], width=bar_width, edgecolor='black', linewidth=0.7)
    plt.bar(x_positions[i], coarsen[i], bottom=uncoarsen[i] + init[i], color=colors[i][2], width=bar_width, edgecolor='black', linewidth=0.7)

# Customize axes
tick_positions = [(i * (5 + group_gap) + 2) for i in range(9)]
plt.xticks(tick_positions, graphs, rotation=30, ha='right', fontsize=170)
plt.yticks(fontsize=150)
plt.ylabel('Speedup', fontsize=170)
# plt.ylim(0, 1.2)  # Set y-axis top to 1
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.6, color='grey')
# Custom legend with software names above
fig = plt.gcf()
legend_elements = []
for i, sw in enumerate(software):
    cmap = gradients[sw]
    legend_elements.append([
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(2), edgecolor='black', label="Uncoarsen"),
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(1), edgecolor='black', label="Init Partition"),
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0), edgecolor='black', label="Coarsen")
    ])

for i, sw in enumerate(software):
    leg = fig.legend(
        handles=legend_elements[i],
        labels=["Uncoarsen", "Init Partition", "Coarsen"],
        loc='upper center',
        bbox_to_anchor=(0.38 + i * 0.34, 2),
        ncol=1, handlelength=1.5, handletextpad=0.5, fontsize=120,
        title=sw, title_fontsize=160
    )
    leg.get_title().set_ha('center')

# Adjust layout
plt.tight_layout(rect=[0, 0.1, 1.9, 1.5])
plt.savefig("figure15.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figure15.png", dpi=400, bbox_inches='tight')
# plt.savefig("three_phase_2.pdf", bbox_inches='tight')
# plt.show()