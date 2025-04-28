import os
import pickle
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from copy import deepcopy

def readComm(n, commFile):
    C = np.zeros((n,), dtype=int)
    x1 = defaultdict(lambda: -1)
    l = 0
    community = []
    comm_from_data = []
    with open(commFile) as file:
        i = 0
        for item in file:
            if item.startswith("#") or item.startswith("%"):
                continue
            x = item.split()
            x2 = int(x[0])
            if x1[x2] == -1:
                x1[x2] = l
                community.append(set([i]))
                comm_from_data.append(x2)
                C[i] = l
                l += 1
                i += 1
            else:
                C[i] = x1[x2]
                community[x1[x2]].add(i)
                i += 1
    return C, community, comm_from_data

def read_mtx(fname):
    G = nx.DiGraph()
    with open(fname) as file:
        i = 0
        for item in file:
            if item.startswith("#") or item.startswith("%"):
                continue
            if i == 0:
                i += 1
                x = item.split()
                n, m = int(x[0]), int(x[2])
            else:
                x = item.split()
                G.add_edge(int(x[0])-1, int(x[1])-1)
                G[int(x[0])-1][int(x[1])-1]['weight'] = 1
    return G, n, m

def find_border_vertices(G, C):
    bv = []
    for u in G.nodes():
        for v in G.neighbors(u):
            if C[u] != C[v]:
                bv.append(u)
                break
    return bv

def compute_LC(G, C, bv):
    comm_count = [dict() for _ in G.nodes]
    number_of_unique_comm = [0 for _ in G.nodes]
    for v in bv:
        lc = [C[x] for x in G.neighbors(v)]
        unique_comms = set(lc)
        number_of_unique_comms = len(unique_comms)
        count = Counter(lc)
        comm_count[v] = count
        number_of_unique_comm[v] = number_of_unique_comms
    return comm_count, number_of_unique_comm

def AWCC_modified(G, S, C, deg):
    v_ngbr_comm = {}
    for v in S:
        if v not in G.nodes():
            continue
        v_ngbr_comm[v] = [C[w] for w in G.neighbors(v)]
    score = 0
    for v in S:
        if v not in G.nodes():
            continue
        try:
            score += len(set(v_ngbr_comm[v])) / (deg[v] + 1)
        except:
            score += 0
    score = score / len(S)
    return score

def edge_removal(G, pcnt_removal, m):
    how_many = int(m*pcnt_removal/100)
    edges = list(G.edges())
    to_remove = np.random.choice([x for x in range(len(edges))], size = how_many)
    edges_to_remove = [edges[x] for x in to_remove]
    
    G.remove_edges_from(edges_to_remove)
    
    return G

def process_graph(graph_file_name, shs_lists, methods):
    pltname = graph_file_name.split(".")[0]
    pickle_filename = f"{pltname}awcc_lists_EdgeRemoval.pkl"

    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as file:
            awcc_lists = pickle.load(file)
        x_val = len(awcc_lists[0]) - 1
    else:
        G, n, m = read_mtx(graph_file_name)
        commFile = "community_" + pltname + ".txt"
        C, _, _ = readComm(n, commFile)
        bv = find_border_vertices(G, C)
        compute_LC(G, C, bv)
        deg = [G.degree(v) for v in range(n)]
        awcc_lists = [[] for _ in range(len(methods))]
        G_R = deepcopy(G)
        x_val = 15
        for i in range(x_val + 1):
            if i > 0:
                G_R = edge_removal(G_R, 5, m)
            for j, S in enumerate(shs_lists):
                score = AWCC_modified(G_R, S, C, deg)
                awcc_lists[j].append(score)
        with open(pickle_filename, 'wb') as file:
            pickle.dump(awcc_lists, file)

    mark = ['x', '.', 'o', '+', '>']
    colors = sns.color_palette("magma", 5)
    x_values = list(range(0, (x_val * 5 + 1), 5))
    plt.figure(figsize=(8, 6))
    for index, line in enumerate(awcc_lists):
        sns.lineplot(x=x_values[:len(line)], y=line, label=f"{methods[index]}", color=colors[index], marker=mark[index])
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 24, 'axes.titlesize': 20})
    plt.xlabel("Removal Percentage", fontsize=24)
    plt.ylabel("Absolute AWCC", fontsize=24)
    plt.title(f"{pltname}", fontsize=24)
    plt.xticks(range(0, max(x_values) + 1, 10), fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"AbsAWCCChange_edge_removal_{pltname}.pdf", format='pdf', dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    methods = ['AP_BICC', 'HAM', 'HIS', 'RS', 'ABC']
    # replace data if the graphs are changed
    # Convert the vertex IDs to 0-indexed.
    graph_data = {
        "moreno_innovation.mtx": [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            [217, 219, 233, 231, 234, 232, 215, 203, 227, 229, 220, 222, 204, 207, 212, 214, 209, 206, 208, 224, 202, 210, 216, 230, 218],
            [126, 193, 14, 197, 120, 196, 169, 166, 182, 165, 185, 73, 127, 39, 192, 186, 199, 178, 195, 189, 198, 200, 202, 204, 22],
            [0, 3, 2, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34, 37, 38, 40],
            [14, 68, 22, 28, 62, 73, 9, 77, 39, 86, 126, 11, 27, 44, 53, 127, 10, 12, 19, 35, 37, 40, 166, 191, 4]
        ],
        "soc-ANU-residence.edges": [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            [197, 88, 27, 173, 153, 141, 121, 207, 176, 209, 188, 190, 215, 4, 198, 181, 5, 76, 52, 168, 210, 25, 160, 28, 58],
            [183, 150, 114, 175, 45, 46, 91, 137, 168, 162, 167, 204, 146, 134, 72, 112, 97, 40, 143, 75, 80, 185, 26, 172, 95],
            [214, 113, 135, 118, 191, 154, 66, 115, 145, 215, 102, 130, 203, 126, 25, 178, 60, 98, 10, 164, 147, 190, 146, 23, 166],
            [69, 26, 112, 183, 97, 168, 70, 1, 56, 72, 68, 107, 120, 35, 45, 53, 82, 127, 16, 99, 124, 12, 23, 24, 46]
        ],
        "fb-pages-politician.mtx": [
            [0, 1972, 5111, 138, 3053, 1473, 4978, 4523, 2334, 1038, 3923, 3874, 766, 98, 2676, 1547, 3622, 5292, 5455, 2755, 568, 1849, 980, 1578, 4674],
            [4057, 2855, 5073, 295, 3701, 3230, 202, 3731, 2583, 4658, 604, 1003, 4302, 1490, 3015, 5672, 4714, 1622, 2575, 4908, 5006, 5734, 833, 224, 5028],
            [3008, 5416, 1864, 1595, 4602, 1836, 4299, 3236, 3638, 287, 5350, 1507, 1920, 5890, 478, 2106, 594, 2069, 2078, 3295, 3188, 4068, 4380, 3029, 3209],
            [1125, 4875, 4167, 1447, 1131, 135, 928, 5365, 2392, 427, 1506, 1829, 4683, 4349, 1106, 411, 2329, 363, 2390, 1205, 5091, 3447, 1722, 1179, 5248],
            [5800, 3576, 1864, 1324, 1474, 2900, 4395, 3008, 192, 1836, 2143, 1965, 2059, 4602, 191, 2306, 4081, 3387, 5882, 442, 2698, 4032, 5072, 4219, 155]
        ],
        "socfb-Caltech36.mtx": [
            [4, 0, 30, 35, 38, 42, 60, 62, 89, 95, 100, 113, 115, 119, 131, 138, 139, 141, 144, 145, 150, 152, 153, 154, 164],
            [163, 608, 650, 34, 456, 636, 301, 742, 289, 36, 665, 336, 393, 20, 427, 553, 201, 554, 257, 292, 1, 27, 287, 213, 239],
            [708, 89, 686, 354, 154, 430, 26, 489, 444, 562, 718, 417, 187, 569, 493, 566, 113, 744, 322, 17, 749, 160, 536, 688, 472],
            [397, 504, 427, 661, 711, 683, 402, 574, 349, 626, 733, 702, 133, 722, 724, 546, 672, 108, 748, 171, 326, 632, 680, 609, 685],
            [708, 222, 89, 277, 637, 256, 354, 531, 734, 115, 624, 669, 84, 625, 150, 264, 428, 452, 549, 614, 622, 77, 372, 483, 513]
        ]
    }

    for graph_file_name, shs_lists in graph_data.items():
        process_graph(graph_file_name, shs_lists, methods)
