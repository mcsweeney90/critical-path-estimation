#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert STG files to DAG objects and save.
"""

import os, pathlib, dill, re
import networkx as nx
from timeit import default_timer as timer
import sys
sys.path.append('../../')   
from Simulator import Task, DAG

####################################################################################################

# Variables etc used throughout.
size = 1000
src_path = 'original/{}'.format(size)
save_path = '{}'.format(size)
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

####################################################################################################

start = timer()
# Read stg files.
for orig in os.listdir(src_path):    
    if orig.endswith('.stg'):  
        print("\n{}".format(orig))
        G = nx.DiGraph()       
        with open("{}/{}".format(src_path, orig)) as f:
            next(f) # Skip first line.            
            for row in f:
                if row[0] == "#":                   
                    break
                # Remove all whitespace - there is probably a nicer way to do this...
                info = " ".join(re.split("\s+", row, flags=re.UNICODE)).strip().split() 
                # Create task.   
                nd = Task()
                nd.ID = int(info[0])
                if info[2] == '0':
                    G.add_node(nd)
                    continue
                # Add connections to predecessors.
                predecessors = list(n for n in G if str(n.ID) in info[3:])
                for p in predecessors:
                    G.add_edge(p, nd)
        # Convert G to a DAG object. 
        dag_name = orig.split('.')[0]
        D = DAG(G, name=dag_name)    
                          
        # Save DAG.
        with open('{}/{}.dill'.format(save_path, dag_name), 'wb') as handle:
            dill.dump(D, handle)        
        
elapsed = timer() - start     
print("Time taken: {} seconds".format(elapsed))   