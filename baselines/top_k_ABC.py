import time

import argparse
import networkit as nk


# Create the parser
parser = argparse.ArgumentParser(description='Process graph file.')

# Add the arguments
parser.add_argument('-f', '--graphfile', type=str, required=True,
                    help='The path to the graph file')

args = parser.parse_args()

graph_file = args.graphfile
#graph_file = "soc-ANU-residence.metis"

G = nk.graphio.METISGraphReader().read(graph_file)

# print(graph_file)

#set number of threads
nk.setNumberOfThreads(64)


# ApproxBetweenness centrality
start_time = time.perf_counter()
abc = nk.centrality.ApproxBetweenness(G, epsilon=0.1)
abc.run()
end_time = time.perf_counter()
elapsed_time_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
print(f"Approx. Betweenness centrality: Elapsed time: {elapsed_time_ms} milliseconds")
top25abc_ = abc.ranking()[:25]
top25abc = []
for (u, val) in top25abc_:
    top25abc.append(u+1) #making it 1-indexed
print(top25abc)


filename = "shs_multiMethod_" + graph_file.split(".")[0] + ".txt"
with open(filename, 'a') as file:
    file.write("\n\nABC:\n")
    for s in top25abc:
        file.write(f"{s}, ")  # Vertex IDs are 1-indexed


