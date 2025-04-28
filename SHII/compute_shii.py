import numpy as np
from collections import defaultdict
import easygraph as eg
import os
from copy import deepcopy
import argparse
import sys

os.chdir(r'C:\Phd\CUDA test\Test\test 1\RS_new25\SC_AD_code_repo\code_and_data_for_comparison\SHII')

#comm labels are assumed to be integers
def readComm(n, commFile):
    Comm = np.zeros((n+1,), dtype=int) #stores comm label (0-indexed) of all vertices (1-indexed). Comm[0] is not used
    x1 = defaultdict(lambda:-1)
    l = 0 #used to make the comm id 0-indexed
    community = [] #stores list of sets of vertices in each comm. E.g., {{1,4,...},...,{33,35,..}}
    comm_from_data = []
    with open(commFile) as file:
        i = 1 #as vertex ids are 1-indexed
        for item in file:
            if item.startswith("#"):
                continue
            if item.startswith("%"):
                continue
            else:
                x = item.split()
                x2 = int(x[0])
                if x1[x2] == -1:
                    x1[x2] = l
                    community.append(set([i]))
                    comm_from_data.append(x2)
                    Comm[i] = l
                    l = l + 1
                    i = i + 1
                else:
                    Comm[i] = x1[x2]
                    community[x1[x2]].add(i)
                    i = i + 1
    return (Comm, community, comm_from_data)

def mtx_to_easygreaph(fname):
    G = eg.Graph() #easygraph graph is 1-indexed
    with open(fname) as file:
        i = 0
        for item in file:
            if item.startswith("#"):
                continue
            if item.startswith("%"):
                continue
            if i == 0:
                i = i + 1
                x = item.split()
                n = int(x[0])
                m = int(x[2])
            else:
                x = item.split()
                G.add_edge(int(x[0]),int(x[1])) ##do +1 if the graph is edgelist or .dat as G accepts 1-indexed vertex id
    return G, n, m

def edge_removal(G, pcnt_removal):
    how_many = int(len(G.edges())*pcnt_removal/100)
    edges = list(G.edges())
    to_remove = np.random.choice([x for x in range(len(edges))], size = how_many)
    edges_to_remove = [edges[x] for x in to_remove]
    
    G.remove_edges(edges_to_remove)
    
    return G

def node_removal(G, pcnt_removal):
    how_many = int(len(G.nodes())*pcnt_removal/100)
    nodes = list(G.nodes())
    to_remove = np.random.choice([x for x in range(len(nodes))], size = how_many)
    nodes_to_remove = [nodes[x] for x in to_remove]
    
    G.remove_nodes(nodes_to_remove)
    
    return G
 
def edge_removal_pref(G, pcnt_removal):
    how_many = int(len(G.edges())*pcnt_removal/100)
    edges = list(G.edges())

    degree_edges = []
    for (u,v) in edges:
        degree_edges.append(G.degree(u)*G.degree(v))
    """
    df = pd.DataFrame()
    df['edges'] = edges
    df['degree_edges'] = degree_edges
    """
    p_list = []
    sum_degree = np.sum(degree_edges)
    for x in degree_edges:
        p_list.append(x/sum_degree)
    
    to_remove = np.random.choice([x for x in range(len(edges))], size = how_many, replace=False, p = p_list)
    edges_to_remove = [edges[x] for x in to_remove]
    
    G.remove_edges(edges_to_remove)
    
    return G

def node_removal_pref(G, pcnt_removal):
    how_many = int(len(G.nodes())*pcnt_removal/100)
    nodes = list(G.nodes())

    degree_nodes = []
    for n in nodes:
        degree_nodes.append(G.degree(n))
    
    p_list = []
    sum_degree = np.sum(degree_nodes)
    for x in degree_nodes:
        p_list.append(x/sum_degree)
        
    to_remove = np.random.choice([x for x in range(len(nodes))], size = how_many, replace=False, p = p_list)
    nodes_to_remove = [nodes[x] for x in to_remove]
    
    G.remove_nodes(nodes_to_remove)
    
    return G    

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="input file", required=True)
args = parser.parse_args()

if os.stat(args.filename).st_size == 0:
    print(f"Error: File '{args.filename}' is empty.\n")
    parser.print_help()
    sys.exit(1)   
graph_file_name =  args.filename


filename = "shs_multiMethod_" + graph_file_name.split(".")[0] + ".txt"

HIS = []
RS = []
HAM = []
AP_BICC = []
ABC = []

with open(filename, 'r') as f:
    lines = f.readlines()

# Process lines
current_list = None

for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith('HIS:'):
        current_list = HIS
        line = line[4:].strip()
    elif line.startswith('RS:'):
        current_list = RS
        line = line[3:].strip()
    elif line.startswith('HAM:'):
        current_list = HAM
        line = line[4:].strip()
    elif line.startswith('AP_BICC:'):
        current_list = AP_BICC
        line = line[8:].strip()
    elif line.startswith('ABC:'):
        current_list = ABC
        line = line[4:].strip()

    # If there is number content in the line
    if line and current_list is not None:
        numbers = [int(x.strip()) for x in line.split(',') if x.strip()]
        current_list.extend(numbers)






# graph_file_name = "moreno_innovation.mtx"


shs_lists = [AP_BICC, HAM, HIS, RS, ABC]
methods = ['AP_BICC', 'HAM', 'HIS', 'RS', 'ABC']


G, n, m = mtx_to_easygreaph(graph_file_name)
commFile = "community_" + graph_file_name.split(".")[0] + ".txt"
Comm, communities, comm_from_data = readComm(n, commFile)

diffusion_models = ['IC','LT'] #use Independent Cascade ('IC') or Linear Threshold ('LT')

C = {frozenset(value) for value in communities}

C_list_of_lists = [list(subset) for subset in C]


avg_shii_lists = [[] for _ in range(len(methods))]

#for itr in range(20):
removal = []
G_R = deepcopy(G)    
filename = "avg_shii_" + graph_file_name.split(".")[0] + ".txt" 

for df_mod in diffusion_models:
    j = 0
    for S in shs_lists:
        shii =  eg.structural_hole_influence_index(G_R, S, C_list_of_lists, df_mod, seedRatio=0.005, Directed=False, randSeedIter=10, countIterations=100)
        avg_shii = sum(shii.values()) / len(shii)
        with open(filename, 'a') as file:
            file.write(f"\n diffusion_model: {df_mod} method: {methods[j]}, avg_shii: {avg_shii}")
        j = j + 1



