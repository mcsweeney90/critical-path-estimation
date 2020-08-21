#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and draw simple example DAG used in write up.
"""

import dill, pathlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout
import sys
sys.path.append('../') 
from Simulator import Task, DAG, Platform, HEFT, PEFT

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

path = 'results/example'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

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

# =============================================================================
# Create a random topology.
# =============================================================================
    
# G1 = nx.gnp_random_graph(5, 0.5, directed=True)
# G2 = nx.DiGraph([(u,v) for (u,v) in G1.edges() if u<v])
# dag = convert_from_nx_graph(G2)
# Save DAG.
# with open('{}/example_dag.dill'.format(path), 'wb') as handle:
#     dill.dump(dag, handle) 

# =============================================================================
# Find suitable costs.
# =============================================================================
# with open('{}/example_dag.dill'.format(path), 'rb') as file:
#     dag = dill.load(file) 
# nw = 2
# platform = Platform(nw, name="{}P".format(nw)) 
# for _ in range(1000):
#     dag.set_costs(platform, method="diagram")
#     mst = dag.minimal_serial_time()
#     p_list, ranks = dag.critical_path_priorities(return_ranks=True)
#     heft_cp_est = ranks[dag.top_sort[0]]
#     heft_mkspan = HEFT(dag, platform, priority_list=p_list)
#     if heft_mkspan > heft_cp_est:
#         continue
#     elif heft_mkspan > mst:
#         continue
#     fulk_mkspan = HEFT(dag, platform, cp_type="F")
#     wm_mkspan = HEFT(dag, platform, cp_type="HEFT", avg_type="WM")
#     opt_mkspan = HEFT(dag, platform, cp_type="optimistic")
#     if fulk_mkspan < heft_mkspan and wm_mkspan < heft_mkspan and opt_mkspan < heft_mkspan:
#         with open('{}/example_dag_with_costs.dill'.format(path), 'wb') as handle:
#             dill.dump(dag, handle)
#         break
    
# =============================================================================
# Draw multiple cost graph.
# =============================================================================
    
nw = 2
platform = Platform(nw, name="{}P".format(nw))        
# Load DAG.
with open('{}/example_dag_with_costs.dill'.format(path), 'rb') as file:
    dag = dill.load(file)     

info, node_weights, edge_weights = {}, {}, {}
for t in dag.top_sort:
    info[t.ID] = list(s.ID for s in dag.graph.successors(t))
    node_weights[t.ID] = tuple(t.comp_costs[w.ID] for w in platform.workers)
    for s in dag.graph.successors(t):        
        costs = [0]
        for w in platform.workers:
            for v in platform.workers:
                if w.ID == v.ID:
                    continue
                costs.append(t.comm_costs[s.ID][(w.ID, v.ID)])
        edge_weights[(t.ID, s.ID)] = tuple(costs)
D = nx.DiGraph()
for n, kids in info.items():
    for c in kids:
        D.add_edge(n, c)
plt.clf()
pos = graphviz_layout(D, prog='dot')    
nx.draw_networkx_nodes(D, pos, node_color='#E5E5E5', node_size=500, alpha=1.0)
nx.draw_networkx_edges(D, pos, width=1.0, snap=True)
nx.draw_networkx_labels(D, pos, font_size=12, font_color='#348ABD', font_weight='bold')
nx.draw_networkx_edge_labels(D, pos, font_size=9, edge_labels=edge_weights, font_color='#E24A33', font_weight='bold')
alt_pos = {}
for p in pos:
    if p == 0:
        alt_pos[p] = (pos[p][0], pos[p][1] + 16)
    elif p == 1 or p == 2:
        alt_pos[p] = (pos[p][0] + 2, pos[p][1] + 16)
    elif p == 3 or p == 5:
        alt_pos[p] = (pos[p][0] - 2, pos[p][1] + 16)
    elif p == 4:
        alt_pos[p] = (pos[p][0] + 13, pos[p][1])
    elif p == 6:        
        alt_pos[p] = (pos[p][0] + 13, pos[p][1] - 4)            
nx.draw_networkx_labels(D, alt_pos, node_weights, font_size=9, font_color='#E24A33', font_weight='bold')
plt.axis("off")     
plt.savefig('{}/simple_graph.png'.format(path), bbox_inches='tight') 

# =============================================================================
# Draw fixed-cost version used in HEFT.
# =============================================================================

# nw = 2
# platform = Platform(nw, name="{}P".format(nw))        
# # Load DAG.
# with open('{}/example_dag_with_costs.dill'.format(path), 'rb') as file:
#     dag = dill.load(file)     

# info, node_weights, edge_weights = {}, {}, {}
# for t in dag.top_sort:
#     info[t.ID] = list(s.ID for s in dag.graph.successors(t))
#     node_weights[t.ID] = t.average_cost()
#     for s in dag.graph.successors(t):  
#         edge_weights[(t.ID, s.ID)] = t.average_comm_cost(s)
# D = nx.DiGraph()
# for n, kids in info.items():
#     for c in kids:
#         D.add_edge(n, c)
# plt.clf()
# pos = graphviz_layout(D, prog='dot')    
# nx.draw_networkx_nodes(D, pos, node_color='#E5E5E5', node_size=500, alpha=1.0)
# nx.draw_networkx_edges(D, pos, width=1.0)
# nx.draw_networkx_labels(D, pos, font_size=12, font_color='#348ABD', font_weight='bold')
# nx.draw_networkx_edge_labels(D, pos, font_size=12, edge_labels=edge_weights, font_color='#E24A33', font_weight='bold')
# alt_pos = {}
# for p in pos:
#     if p == 0:
#         alt_pos[p] = (pos[p][0], pos[p][1] + 16)
#     elif p == 1 or p == 2:
#         alt_pos[p] = (pos[p][0], pos[p][1] + 16)
#     elif p == 3 or p == 5:
#         alt_pos[p] = (pos[p][0], pos[p][1] + 16)
#     elif p == 4:
#         alt_pos[p] = (pos[p][0] + 12, pos[p][1])
#     elif p == 6:        
#         alt_pos[p] = (pos[p][0] + 12, pos[p][1] - 4)            
# nx.draw_networkx_labels(D, alt_pos, node_weights, font_size=12, font_color='#E24A33', font_weight='bold')
# plt.axis("off")     
# plt.savefig('{}/simple_graph_fixed.png'.format(path), bbox_inches='tight')

# =============================================================================
# Compute ranks. 
# =============================================================================

# nw = 2
# platform = Platform(nw, name="{}P".format(nw))        
# # Load DAG.
# with open('{}/example_dag_with_costs.dill'.format(path), 'rb') as file:
#     dag = dill.load(file)

# mst = dag.minimal_serial_time()
# print("MST = {}".format(mst))
# p_list, ranks = dag.critical_path_priorities(return_ranks=True)
# print("\nHEFT task ranks: {}".format({k.ID:v for k, v in ranks.items()}))
# heft_mkspan = HEFT(dag, platform, priority_list=p_list)
# print("HEFT makespan: {}".format(heft_mkspan))
# opt_list, ranks = dag.critical_path_priorities(cp_type="optimistic", return_ranks=True)
# print("\nOpt task ranks: {}".format({k.ID:v for k, v in ranks.items()}))
# opt_mkspan = HEFT(dag, platform, priority_list=opt_list)
# print("Opt makespan: {}".format(opt_mkspan))
# fulk_list, ranks = dag.critical_path_priorities(cp_type="F", return_ranks=True)
# print("\nFulk task ranks: {}".format({k.ID:v for k, v in ranks.items()}))
# fulk_mkspan = HEFT(dag, platform, priority_list=fulk_list)
# print("Fulk makespan: {}".format(fulk_mkspan))


# =============================================================================
# Monte Carlo ranks.
# =============================================================================

# nw = 2
# platform = Platform(nw, name="{}P".format(nw))        
# # Load DAG.
# with open('{}/example_dag_with_costs.dill'.format(path), 'rb') as file:
#     dag = dill.load(file)

# nsamples = [1, 10, 20, 50, 100, 1000]

# for ns in nsamples:
#     p_list, ranks = dag.critical_path_priorities(cp_type="MC", mc_samples=ns, return_ranks=True)
#     print("\nMC{} task ranks: {}".format(ns, {k.ID:v for k, v in ranks.items()}))
#     # mkspan = HEFT(dag, platform, priority_list=p_list)
#     # print("MC{} makespan: {}".format(ns, mkspan))
    
# =============================================================================
# PEFT.
# =============================================================================
    
# nw = 2
# platform = Platform(nw, name="{}P".format(nw))        
# # Load DAG.
# with open('{}/example_dag_with_costs.dill'.format(path), 'rb') as file:
#     dag = dill.load(file)
    
# mst = dag.minimal_serial_time()
# print("MST = {}".format(mst))
    
# OCT = dag.optimistic_cost_table()
# print(OCT)
# with open("{}/peft_schedule.txt".format(path), "w") as dest:
#     peft_mkspan = PEFT(dag, platform, schedule_dest=dest)
#     print(peft_mkspan)
    
# =============================================================================
# Testing.
# =============================================================================
    
# fig, ax = plt.subplots(dpi=400)
# ax.broken_barh([(110, 30), (150, 10)], (2, 10), facecolors='#E24A33')
# ax.broken_barh([(10, 50), (100, 20), (130, 10)], (14, 10),
#                facecolors='#E24A33')
# ax.set_ylim(0, 27)
# ax.set_xlim(0, 200)
# ax.set_xlabel('Time units')
# ax.set_yticks([15, 25])
# ax.set_yticklabels(['P1', 'P2'])
# ax.grid(False)

# plt.show()
    
# nw = 2
# platform = Platform(nw, name="{}P".format(nw))        
# # Load DAG.
# with open('{}/example_dag_with_costs.dill'.format(path), 'rb') as file:
#     dag = dill.load(file)

# heft_mkspan = HEFT(dag, platform, schedule_img_dest=path)
# print("HEFT makespan: {}".format(heft_mkspan))