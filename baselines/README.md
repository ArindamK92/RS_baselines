## Run all baseline code

Run HIS Code:    
-----------------  
The implementation is taken from the paper:
T. Lou and J. Tang. 2013. "Mining structural hole spanners through information
diffusion in social networks". In Proceedings of the 22nd international conference
on World Wide Web. 825–836

use -c to add community_file (default is "community.txt")  
add -a to specify the communities to consider  (use the top communities)
use -g to specify the graph file in .dat format 

```
g++ -o op_HIS HIS.cpp
./op_HIS -g graphFile.dat -c communityfile.txt -a 1 -a 2 -a 8
```


Run other baselines:  
---------------------  
Required libraries: 
easygraph https://easy-graph.github.io/docs/install.html  
networkx https://networkx.org/documentation/stable/install.html  
numpy https://numpy.org/install/  
scipy https://scipy.org/install/  
networkit https://networkit.github.io/  
 
```
python top_k_multiBaseline.py -f graphFile.mtx
```

Convert graph to .metis format for ABC baseline.  
```
g++ -O3 -o op_mtx2metis mtxToMetis_skipLines.cpp  
./op_mtx2metis graphFile.mtx > graphFile.metis  
```

Find top K vertices with highest ABC 
```
python top_k_ABC.py -f twitter.metis  
```
