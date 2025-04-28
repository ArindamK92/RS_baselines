import networkx as nx
import numpy as np
from collections import defaultdict
import os
from scipy.stats import entropy
import argparse
import sys
from collections import Counter
from copy import deepcopy
import easygraph as eg

def readComm(n, commFile):
    C = np.zeros((n,), dtype=int) #stores label of all vertices
    x1 = defaultdict(lambda:-1)
    l = 0
    community = []
    comm_from_data = []
    with open(commFile) as file:
        i = 0
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
                    C[i] = l
                    l = l + 1
                    i = i + 1
                else:
                    C[i] = x1[x2]
                    community[x1[x2]].add(i)
                    i = i + 1
    return (C, community, comm_from_data)

#comm labels are assumed to be integers
def readComm2(n, commFile):
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

def read_mtx(fname):
    G = nx.DiGraph()
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
                G.add_edge(int(x[0])-1,int(x[1])-1) ##do -1 if the .mtx graph. Don't do -1 if edgelist or .dat
                G[int(x[0])-1][int(x[1])-1]['weight'] = 1
    return G, n, m

def find_border_vertices(G, C):
    bv = [] # community border vertices
    for u in G.nodes():
        for v in G.neighbors(u):
            if C[u] != C[v]:
                bv.append(u)
                break
    return bv


def compute_LC(G, C, bv):
    comm_count = [dict() for v in G.nodes] # ith element stores a dictionay containing the count of ngbrs(of node i) connected to different communities
    number_of_unique_comm = [0 for v in G.nodes] # ith element stores the number of unique communities connected to i through the neighbors
    for v in bv:
        lc = [] # list of neighbor communities
        for x in G.neighbors(v):
            lc.append(C[x])
        unique_comms = set(lc)
        number_of_unique_comms = len(unique_comms) # number of unique communities connected to v
        count = Counter(lc) # count of neighbors from each community. count[1] gives the ngbr count from comm 1
        comm_count[v] = count
        number_of_unique_comm[v] = number_of_unique_comms
    return comm_count, number_of_unique_comm
        


if __name__ == "__main__":
    G = nx.Graph() # Directed graph
    n = 0 #number of vertices
    m = 0 #number of edges
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="input file", required=True)
    args = parser.parse_args()
    
    if os.stat(args.filename).st_size == 0:
        print(f"Error: File '{args.filename}' is empty.\n")
        parser.print_help()
        sys.exit(1)   
    graph_file_name =  args.filename
    
    #os.chdir(r'E:\Research\Large Dynamic Network\new\code')
    # graph_file_name = "sampled_socfb-Caltech36.mtx"
    G, n, m = read_mtx(graph_file_name)
    commFile = "community_" + graph_file_name.split(".")[0] + ".txt"
    C, communities, comm_from_data = readComm(n, commFile)
    filename = "shs_multiMethod_" + graph_file_name.split(".")[0] + ".txt"
    
    bv = find_border_vertices(G, C) # community border vertices. They have the potential to be a spanner
    #ngbr_of_bv = find_ngbrs_of_all_bv(G, bv) # neighbors set of all bv
    
    comm_count, number_of_unique_comm = compute_LC(G, C, bv)
    
    G_back = deepcopy(G)
    
    
    #G_1 = nx.Graph()
    #W = dict()
    Adj = np.zeros((n,n))
    for u in bv:
        #compute weight (required for triad type 1)
        for v in G.neighbors(u):
            cx = comm_count[v].copy()
            if len(cx) == 0:
                continue
            del cx[C[u]]  #we remove the count for C[u] as we are searching for other communities
            mod_L = sum(cx.values()) # |L(u,v,C[u])|
            if mod_L == 0:
                continue
            unique_comm = len(set(cx.values()))
            P = [value/mod_L for value in cx.values()]
            H = entropy(P, base=2)
            weight = H * mod_L
            if weight > 0:
                # G_1.add_edge(u, v)
                # W[(u,v)] = weight
                Adj[u][v] = weight
        #compute self weight (required for triad type 2)
        cx1 = comm_count[u].copy()
        del cx1[C[u]]  #we remove the count for C[u] as we are searching for other communities
        mod_L = sum(cx1.values()) # |L(u,v,C[u])|
        P = [value/mod_L for value in cx1.values()]
        H = entropy(P, base=2)
        weight = H * mod_L
        if weight > 0:
            Adj[u][u] = weight
        
    
    # compute clustering coefficients
    max_weight = np.max(Adj)
    Adj  = Adj / max_weight  # we normalize the weights
    #coeff = np.zeros(n)
    coeff = {}
    for u in bv:
        #triples = 0
        triangles = 0
        for v in bv:
            for w in bv:
                if (v == w) or (v == u) or (w == u):
                    continue;
                if C[v] == C[w]:
                    continue
                if Adj[u][w] > 0 and Adj[v][w] > 0:
                    #triples += 1
                    if Adj[u][v] > 0 and C[u] != C[w]: #Triad type 1. we consider (u,v,w) and it covers (u,w,v) also due to 2 for loops
                        triangles += (Adj[u][v] * Adj[u][w] * Adj[v][w]) ** (1/3)
                        #print("triad1::",u,":",v,":",w)
                    elif C[u] == C[v]: #C[u] != C[w] is already satisfied due to check at line 226 C[v] != C[w]
                        triangles += (Adj[v][v] * Adj[u][w] * Adj[v][w]) ** (1/3)
                        #print("triad2::",u,":",v,":",w)
                
        
        coeff[u] = triangles / (G.degree[u] * (G.degree[u] - 1)) if G.degree[u] >= 2 else 0
    
    
    ##Get top k robust spanners
    sorted_coeff = dict(sorted(coeff.items(), key=lambda item: item[1], reverse=True))
    top_k = list(sorted_coeff.keys())[:25]
    print(top_k)
    
    with open(filename, 'a') as file:
        file.write("\n\nRS:\n")
        for s in top_k:
            file.write(f"{s+1}, ") # we print 1-indexed vertex ids
            
            
    ## HAM and APBICC      
    G, n, m = mtx_to_easygreaph(graph_file_name)
    commFile = "community_" + graph_file_name.split(".")[0] + ".txt"
    Comm, communities, comm_from_data = readComm2(n, commFile)
    filename = "shs_multiMethod_" + graph_file_name.split(".")[0] + ".txt"
    diffusion_model = 'LT' #use Independent Cascade ('IC') or Linear Threshold ('LT')
    k = 25 # top k spanner
    
    ##for HAM
    # PARAMETERS:
    # G (easygraph.Graph) – An undirected graph.
    # k (int) – top - k structural hole spanners
    # c (int) – the number of communities
    # ground_truth_labels (list of lists) – The label of each node’s community.
    # RETURNS:
    # top_k_nodes (list) – The top-k structural hole spanners.
    # SH_score (dict) – The structural hole spanners score for each node, given by HAM.
    # cmnt_labels (dict) – The communities label of each node.

    # Now, create the desired list
    C_single_list = [[Comm[v]] for v, val in G.nodes.items()]


    shs5 = eg.get_structural_holes_HAM(G,
                            k = k, # To find top k structural holes spanners.
                             c = len(communities), #number of communities
                             ground_truth_labels = C_single_list # The ground truth labels for each node - community detection result, for example.
                            )
    with open(filename, 'a') as file:
        file.write("\n\nHAM:\n")
        for s in shs5[0]:
            file.write(f"{s}, ")    # Vertex IDs are already 1-indexed

    shs6 = eg.AP_BICC(G,k=k,K=int(n/4),l=4) 


    with open(filename, 'a') as file:
        file.write("\n\nAP_BICC:\n")
        for s in shs6:
            file.write(f"{s}, ")  # Vertex IDs are already 1-indexed

            