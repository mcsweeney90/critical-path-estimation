#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and save Cholesky DAGs.
"""

import dill, pathlib
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import sys
sys.path.append('../../') 
from Simulator import Task, DAG, Platform, HEFT, PEFT, CPOP

def cholesky(n_tiles, draw=False):
    """
    Create a DAG object representing a tiled Cholesky factorization of a matrix.
    
    Parameters
    ------------------------    
    num_tiles - int
    The number of tiles along each axis of the matrix.  
    
    draw - bool
    If True, save an image of the DAG using dag.draw_graph. 
    
    Returns
    ------------------------
    dag - DAG object (see Graph.py module)
    Represents the Cholesky factorization task DAG.   
    """
    
    last_acted_on = {} # Useful for keeping track of things...
    
    G = nx.DiGraph()
    
    for k in range(n_tiles): # Grow the DAG column by column.     
        
        node1 = Task(task_type="POTRF")
        if k > 0:
            # Find the task which last acted on the tile.            
            for node in G: 
                if last_acted_on[(k, k)] == node:
                    G.add_edge(node, node1)
                    break     
        last_acted_on[(k, k)] = node1                                    
        
        for i in range(k + 1, n_tiles):
            node2 = Task(task_type="TRSM")
            G.add_edge(node1, node2)
            try:
                for node in G:
                    if last_acted_on[(i, k)] == node:
                        G.add_edge(node, node2)
                        break
            except KeyError:
                pass
            last_acted_on[(i, k)] = node2            
            
        for i in range(k + 1, n_tiles): 
            node3 = Task(task_type="SYRK")
            try:
                for node in G:
                    if last_acted_on[(i, i)] == node:
                        G.add_edge(node, node3)
                        break
            except KeyError:
                pass
            last_acted_on[(i, i)] = node3
            
            try:
                for node in G:
                    if last_acted_on[(i, k)] == node:
                        G.add_edge(node, node3)
                        break
            except KeyError:
                pass
                
            for j in range(k + 1, i):               
                node4 = Task(task_type="GEMM") 
                try:
                    for node in G:
                        if last_acted_on[(i, j)] == node:
                            G.add_edge(node, node4)
                            break
                except KeyError:
                    pass
                last_acted_on[(i, j)] = node4
                
                try:
                    for node in G:
                        if last_acted_on[(i, k)] == node:
                            G.add_edge(node, node4)
                            break
                except KeyError:
                    pass
                
                try:
                    for node in G:
                        if last_acted_on[(j, k)] == node:
                            G.add_edge(node, node4)
                            break
                except KeyError:
                    pass   
                
    # Create the DAG object. 
    dag = DAG(G, name="cholesky")  
    if draw:
        dag.draw_graph()    
    return dag 

# =============================================================================
# Save the DAGs for nb = 32, 64, 128, 256, 512, 1024.
# =============================================================================

# start = timer()

# # Set up the target platforms.
# multiple = Platform(8, name="Single_GPU")

# # Create and save the actual DAG objects.
# for nb in [128, 1024]:
#     print(nb)
#     for adt in ["no_adt", "perfect_adt"]:
#         save_path = 'nb{}/{}'.format(nb, adt)
#         pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
#         # Load timing data.
#         with open('skylake_V100_cost_samples/{}/nb{}.dill'.format(adt, nb), 'rb') as file:
#             comp_data, comm_data = dill.load(file)
#         comp_costs = {"C" : {}, "G" : {}}
#         comm_costs = {"CC" : {}, "CG" : {}, "GC" : {}, "GG" : {}}         
#         for kernel in ["GEMM", "POTRF", "SYRK", "TRSM"]:
#             comp_costs["C"][kernel] = np.mean(comp_data["C"][kernel])
#             comp_costs["G"][kernel] = np.mean(comp_data["G"][kernel])
#             comm_costs["CC"][kernel] = np.mean(comm_data["CC"][kernel])
#             comm_costs["CG"][kernel] = np.mean(comm_data["CG"][kernel])
#             comm_costs["GC"][kernel] = np.mean(comm_data["GC"][kernel])
#             comm_costs["GG"][kernel] = np.mean(comm_data["GG"][kernel])        
            
#         for nt in range(5, 11, 5):  
#             print(nt)
#             # Construct DAG topology.
#             dag = cholesky(n_tiles=nt)            
#             # Set task costs.
#             for task in dag.graph:
#                 # Computation.
#                 for i, w in enumerate(multiple.workers):
#                     if i < 7:
#                         task.comp_costs[w.ID] = comp_costs["C"][task.type]
#                     else:
#                         task.comp_costs[w.ID] = comp_costs["G"][task.type]
#                 # Communication.
#                 for child in dag.graph.successors(task):#
#                     task.comm_costs[child.ID] = {}
#                     for u in range(multiple.n_workers):
#                         for v in range(multiple.n_workers):
#                             if u == v:
#                                 task.comm_costs[child.ID][("P{}".format(u), "P{}".format(v))] = 0.0
#                             elif u < 7 and v < 7:
#                                 task.comm_costs[child.ID][("P{}".format(u), "P{}".format(v))] = comm_costs["CC"][child.type]
#                             elif u < 7 and v > 6:
#                                 task.comm_costs[child.ID][("P{}".format(u), "P{}".format(v))] = comm_costs["CG"][child.type]
#                             elif u > 6 and v < 7:
#                                 task.comm_costs[child.ID][("P{}".format(u), "P{}".format(v))] = comm_costs["GC"][child.type]
#                             elif u > 6 and v > 6:
#                                 task.comm_costs[child.ID][("P{}".format(u), "P{}".format(v))] = comm_costs["GG"][child.type]
                            
#             # Update DAG attributes.
#             dag.costs_set = True
#             dag.target_platform = "Single_GPU"
#             dag.name += "_nb{}_nt{}".format(nb, dag.n_tasks)
#             with open('{}/{}tasks.dill'.format(save_path, dag.n_tasks), 'wb') as handle:
#                 dill.dump(dag, handle)  
# elapsed = timer() - start     
# print("Time taken to create and save DAGs: {} minutes.".format(elapsed / 60))     
        
# =============================================================================
# Save human-readable summaries of all the DAGs.
# =============================================================================

nb = 128
nt = 35
adt = "perfect_adt"
with open('nb{}/{}/{}tasks.dill'.format(nb, adt, nt), 'rb') as file:
    dag = dill.load(file)
dag.print_info()

cp_lengths = dag.critical_paths(cp_type="Fulkerson")
print(cp_lengths)

# single = Platform(8, name="Single_GPU")
# heft_mkspan = HEFT(dag, single)
# print("HEFT makespan: {}".format(heft_mkspan))
# cpop_mkspan = CPOP(dag, single)
# print("CPOP makespan: {}".format(cpop_mkspan))
        
# start = timer()
    
# summary_path = 'summaries'
# pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)

# for nb in [128, 1024]:
#     print(nb)
#     for adt in ["no_adt", "perfect_adt"]:
#         with open("summaries/nb{}_{}.txt".format(nb, adt), "w") as dest: 
#             for nt in [35, 220]:#, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]:
#                 print(nt)
#                 with open('nb{}/{}/{}tasks.dill'.format(nb, adt, nt), 'rb') as file:
#                     dag = dill.load(file)
#                 dag.print_info(filepath=dest)
# elapsed = timer() - start  
   
# print("Time taken to compute and save DAG summaries: {} minutes.".format(elapsed / 60)) 

