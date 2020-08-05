#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anything that needs to be tested...
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout
import sys
sys.path.append('../') 
from Simulator import Task, DAG, Platform, HEFT, PEFT, CPOP

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.

####################################################################################################

def convert_from_nx_graph(graph, single_root=True, single_exit=True):
    """ 
    Creates a DAG object from an input DiGraph.
    """
    # Make the graph directed if it isn't already.
    if graph.is_directed():
        G = graph
    else:
        G = nx.DiGraph()
        G.name = graph.name
        G.add_nodes_from(graph)    
        done = set()
        for u, v in graph.edges():
            if (v, u) not in done:
                G.add_edge(u, v)
                done.add((u, v))  
        G.graph = deepcopy(graph.graph)     
    # Look for cycles...
    try:
        nx.topological_sort(G)
    except nx.NetworkXUnfeasible:
        raise ValueError('Input graph in convert_from_nx_graph has at least one cycle so is not a DAG!')
    
     # Add single exit node if desired.
    if single_root:
        roots = set(nd for nd in G if not list(G.predecessors(nd)))
        num_roots = len(roots)
        if num_roots > 1:
            source = -1
            G.add_node(source)
            for nd in G:
                if nd in roots:
                    G.add_edge(source, nd)
    if single_exit:
        exits = set(nd for nd in G if not list(G.successors(nd)))
        num_exits = len(exits)
        if num_exits > 1:
            terminus = len(G)
            G.add_node(terminus)
            for nd in G:
                if nd in exits:
                    G.add_edge(nd, terminus) 
    
    # Construct equivalent task graph.
    T = nx.DiGraph()
    done = set()
    for i, t in enumerate(nx.topological_sort(G)):
        if t not in done:           
            nd = Task()                      
            nd.ID = int(t) + 1
            nd.entry = True
            done.add(t)
        else:
            for n in T:
                if n.ID == int(t) + 1:
                    nd = n 
                    break        
        count = 0
        for s in G.successors(t):
            count += 1
            if s not in done:
                nd1 = Task()                               
                nd1.ID = int(s) + 1
                done.add(s) 
            else:
                for n in T:
                    if n.ID == int(s) + 1:
                        nd1 = n
                        break
            T.add_edge(nd, nd1) 
        if not count:
            nd.exit = True              
    return DAG(T, name="rand{}".format(len(G)))
    

G1 = nx.gnp_random_graph(4, 0.5, directed=True)
G2 = nx.DiGraph([(u,v) for (u,v) in G1.edges() if u<v])
dag = convert_from_nx_graph(G2)
nw = 2
platform = Platform(nw, name="{}P".format(nw)) 
dag.set_costs(platform, target_ccr=0.5, method="unrelated", het_factor=2.0)

info, node_weights, edge_weights = {}, {}, {}
for t in dag.top_sort:
    info[t.ID] = list(s.ID for s in dag.graph.successors(t))
    node_weights[t.ID] = tuple(int(t.comp_costs[w.ID]) for w in platform.workers)
    for s in dag.graph.successors(t):        
        costs = [0]
        for w in platform.workers:
            for v in platform.workers:
                if w.ID == v.ID:
                    continue
                costs.append(int(t.comm_costs[s.ID][(w.ID, v.ID)]))
        edge_weights[(t.ID, s.ID)] = tuple(costs)

D = nx.DiGraph()
for n, kids in info.items():
    for c in kids:
        D.add_edge(n, c)

print(len(D))
plt.clf()
pos = graphviz_layout(D, prog='dot')    
nx.draw_networkx_nodes(D, pos, node_color='#E5E5E5', node_size=500, alpha=1.0)
nx.draw_networkx_edges(D, pos, width=1.0)
nx.draw_networkx_labels(D, pos, font_size=12, font_color='#348ABD')
nx.draw_networkx_edge_labels(D, pos, font_size=10, edge_labels=edge_weights, font_color='#E24A33')
alt_pos = {}
for p in pos:
    alt_pos[p] = (pos[p][0], pos[p][1] + 12)            
nx.draw_networkx_labels(D, alt_pos, node_weights, font_size=10, font_color='#E24A33')

plt.axis("off")     
plt.savefig('simple_graph.png', bbox_inches='tight') 

p_list, ranks = dag.critical_path_priorities(return_ranks=True)
print("Upward rank of entry task: {}".format(ranks[dag.top_sort[0]]))
heft_mkspan = HEFT(dag, platform, priority_list=p_list)
print("HEFT makespan: {}".format(heft_mkspan))
fulk_mkspan = HEFT(dag, platform, cp_type="F")
print("Fulk makespan: {}".format(fulk_mkspan))






# Cholesky.
# nb = 128
# nt = 35
# adt = "perfect_adt"
# with open('nb{}/{}/{}tasks.dill'.format(nb, adt, nt), 'rb') as file:
#     dag = dill.load(file)
# dag.print_info()

# STG
# stg_dag_size = 100
# stg_dag_path = '../graphs/STG/{}'.format(stg_dag_size)
# with open('{}/rand0028.dill'.format(stg_dag_path), 'rb') as file:
#     dag = dill.load(file)

# plat = Platform(4, name="4P")  
# dag.set_costs(plat, target_ccr=1.0, method="unrelated", het_factor=2.0)
# dag.print_info()
# heft_mkspan = HEFT(dag, plat, cp_type="HEFT")
# print("HEFT makespan: {}".format(heft_mkspan))
# mc_priority_list = dag.critical_path_priorities(cp_type="WMC", mc_samples=10)
# mcheft_mkspan = HEFT(dag, plat, priority_list=mc_priority_list)
# print("MC makespan: {}".format(mcheft_mkspan))
# rand_list = dag.top_sort
# rand_mkspan = HEFT(dag, plat, priority_list=rand_list)
# print("Rand makespan: {}".format(rand_mkspan))


# weighted_mkspan = HEFT(dag, plat, cp_type="HEFT", avg_type="HEFT-WM")
# print("Weighted makespan: {}".format(weighted_mkspan))
# fulk_mkspan = HEFT(dag, plat, cp_type="Fulkerson")
# print("Fulkerson makespan: {}".format(fulk_mkspan))
# wfulk_mkspan = HEFT(dag, plat, cp_type="Fulkerson", avg_type="HEFT-WM")
# print("Weighted Fulkerson makespan: {}".format(wfulk_mkspan))


# cp_lengths = dag.critical_paths(cp_type="Fulkerson")
# print(cp_lengths)

# single = Platform(8, name="Single_GPU")
# heft_mkspan = HEFT(dag, single)
# print("HEFT makespan: {}".format(heft_mkspan))
# cpop_mkspan = CPOP(dag, single)
# print("CPOP makespan: {}".format(cpop_mkspan))

