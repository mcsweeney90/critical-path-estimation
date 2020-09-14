#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used for drawing DAGs used in the report.
"""
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt 

# =============================================================================
# Simple example graph.
# =============================================================================

def simple_graph():
    """Matplotlib"""
    G = nx.DiGraph()
    
    info = {1 : [2, 3], 2 : [4], 3 : [4]}
    node_weights={1:"(1, 9)", 2:"(1, 9)", 3:"(3, 5)", 4:"(1, 9)"}
    edge_weights = {(1, 2) : (0, 1, 1),
                    (1, 3) : (0, 1, 1),
                    (2, 4) : (0, 1, 1),
                    (3, 4) : (0, 1, 1)}    
    
    for n, kids in info.items():
        for c in kids:
            G.add_edge(n, c)
                
    pos = graphviz_layout(G, prog='dot')
    
    nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 3, 4], node_color='#E5E5E5', node_size=500, alpha=1.0)
    nx.draw_networkx_edges(G, pos, width=1.0)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='#348ABD')
    nx.draw_networkx_edge_labels(G, pos, font_size=10, edge_labels=edge_weights, font_color='#E24A33')
        
    alt_pos = {}
    for p in pos:
        if p == 4:
            alt_pos[p] = (pos[p][0], pos[p][1] - 12)
        else:
            alt_pos[p] = (pos[p][0], pos[p][1] + 12)            
    nx.draw_networkx_labels(G, alt_pos, node_weights, font_size=10, font_color='#E24A33')
    
    plt.axis("off")     
    plt.savefig('simple_graph.png', bbox_inches='tight') 
    
# simple_graph()

# =============================================================================
# Simple example graph.
# =============================================================================

def fixed_simple_graph():
    """Matplotlib"""
    G = nx.DiGraph()
    
    info = {1 : [2, 3], 2 : [4], 3 : [4]}
    node_weights={1:"5", 2:"5", 3:"4", 4:"5"}
    edge_weights = {(1, 2) : "1/2",
                    (1, 3) : "1/2",
                    (2, 4) : "1/2",
                    (3, 4) : "1/2"}    
    
    for n, kids in info.items():
        for c in kids:
            G.add_edge(n, c)
                
    pos = graphviz_layout(G, prog='dot')
    
    nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 3, 4], node_color='#E5E5E5', node_size=500, alpha=1.0)
    nx.draw_networkx_edges(G, pos, width=1.0)
    nx.draw_networkx_edges(G, pos, edgelist=[(1, 2), (2, 4)], edge_color='#E24A33', alpha=0.5, width=8.0)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='#348ABD')
    nx.draw_networkx_edge_labels(G, pos, font_size=10, edge_labels=edge_weights, font_color='#E24A33')
        
    alt_pos = {}
    for p in pos:
        if p == 4:
            alt_pos[p] = (pos[p][0], pos[p][1] - 12)
        else:
            alt_pos[p] = (pos[p][0], pos[p][1] + 12)            
    nx.draw_networkx_labels(G, alt_pos, node_weights, font_size=10, font_color='#E24A33')
    
    plt.axis("off")     
    plt.savefig('fixed_simple_graph.png', bbox_inches='tight') 
    
fixed_simple_graph()

# =============================================================================
# Upward rank vs Fulkerson, fixed-cost version.
# =============================================================================
    
# def expected_node_weight(tc, tg, nc, ng):
#     return (tc*nc + tg*ng) / (nc + ng)

# def expected_edge_weight(e, nc, ng):
#     w = e[1] * nc * (nc - 1)
#     w += (e[2] + e[4]) * nc * ng
#     w += e[3] * ng * (ng - 1)
#     # w /= (nc + ng)**2
#     return w

# def fixed_example_graph():
#     """Matplotlib"""
#     G = nx.DiGraph()
    
#     info = {1 : [2, 3], 2 : [3, 4], 3 : [4]}
#     node_weights={1:3, 2:3, 3:2, 4:3}
#     edge_weights = {(1, 2) : "1/2",
#                     (1, 3) : "19/18",
#                     (2, 3) : "7/9",
#                     (2, 4) : "10/9",
#                     (3, 4) : "8/9"}    
    
#     for n, kids in info.items():
#         for c in kids:
#             G.add_edge(n, c)
                
#     pos = graphviz_layout(G, prog='dot')
    
#     nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 3, 4], node_color='#E5E5E5', node_size=500, alpha=1.0)
#     nx.draw_networkx_edges(G, pos, width=1.0)
#     nx.draw_networkx_edges(G, pos, edgelist=[(1, 2), (2, 3), (3, 4)], edge_color='#E24A33', alpha=0.5, width=8.0)    
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color='#348ABD')
#     nx.draw_networkx_edge_labels(G, pos, font_size=10, edge_labels=edge_weights, font_color='#E24A33')
        
#     alt_pos = {}
#     for p in pos:
#         if p == 3 or p == 4:
#             alt_pos[p] = (pos[p][0], pos[p][1] - 18)
#         else:
#             alt_pos[p] = (pos[p][0], pos[p][1] + 17)            
#     nx.draw_networkx_labels(G, alt_pos, node_weights, font_size=10, font_color='#E24A33')
    
#     plt.axis("off")     
#     plt.savefig('fixed_example_graph.png', bbox_inches='tight') 
    
# # fixed_example_graph()
    
# # =============================================================================
# # Convention for AoA networks.
# # =============================================================================

# def AoA_labels():
#     """
#     Labeling convention for equivalent AoA network.
#     """
#     G = nx.DiGraph()
    
#     info = {'$t_i$' : ['$t_k$']}
#     edge_weights = {('$t_i$', '$t_k$') : r'($\tilde{c}_{ik}$, $\tilde{C}_{ik}^c$, $\tilde{C}_{ik}^g$, $\tilde{g}_{ik}$, $\tilde{G}_{ik}^g$, $\tilde{G}_{ik}^c$)'}    
    
#     for n, kids in info.items():
#         for c in kids:
#             G.add_edge(n, c)
            
#     G.graph['graph'] = {'rankdir':'LR'}
                
#     pos = graphviz_layout(G, prog='dot') 
    
#     nx.draw_networkx_nodes(G, pos, node_color='#E5E5E5', node_size=500, alpha=1.0)
#     nx.draw_networkx_edges(G, pos, width=1.0)
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color='#348ABD')
#     nx.draw_networkx_edge_labels(G, pos, font_size=12, edge_labels=edge_weights, font_color='#E24A33')
            
#     plt.axis("off")     
#     plt.savefig('aoa_labels.png', bbox_inches='tight') 
    
# # AoA_labels()

# # =============================================================================
# # AoA version of simple example DAG.
# # =============================================================================

# def AoA_example_graph():
#     """AoA version of simple example graph."""
#     G = nx.DiGraph()
    
#     info = {1 : [2, 3], 2 : [3, 4], 3 : [4]}
#     edge_weights = {(1, 2) : (1, 1, 2, 7, 8, 8),
#                     (1, 3) : (1, 2, 3, 7, 8, 8),
#                     (2, 3) : (2, 2, 3, 5, 7, 7),
#                     (2, 4) : (5, 6, 6, 8, 10, 10),
#                     (3, 4) : (4, 5, 5, 7, 9, 8)}    
    
#     for n, kids in info.items():
#         for c in kids:
#             G.add_edge(n, c)
                
#     pos = graphviz_layout(G, prog='dot')
    
#     nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 3, 4], node_color='#E5E5E5', node_size=500, alpha=1.0)
#     nx.draw_networkx_edges(G, pos, width=1.0)
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color='#348ABD')
#     nx.draw_networkx_edge_labels(G, pos, font_size=10, edge_labels=edge_weights, font_color='#E24A33')
            
#     plt.axis("off")     
#     plt.savefig('aoa_example_graph.png', bbox_inches='tight') 
        
# # AoA_example_graph()
    
# # =============================================================================
# # Expected value version of AoA example graph.
# # =============================================================================

# # nc, ng = 4, 2 
# # d = 1 #(nc + ng)**2
# # edge_probs = [nc/d, (nc * (nc - 1))/d, (nc * ng) / d,
# #               ng/d, (ng * (ng - 1))/d, (nc * ng)/ d]
# # print(sum(e[0]*e[1] for e in zip((4, 5, 5, 7, 9, 8), edge_probs)))
    
# def fixed_AoA_example_graph():
#     """AoA version of simple example graph."""
#     G = nx.DiGraph()
    
#     info = {1 : [2, 3], 2 : [3, 4], 3 : [4]}
#     edge_weights = {(1, 2) : "21/6",
#                     (1, 3) : "73/18",
#                     (2, 3) : "34/9",
#                     (2, 4) : "64/9", # TODO: check, got edges mixed up at one point.
#                     (3, 4) : "53/9"}    
    
#     for n, kids in info.items():
#         for c in kids:
#             G.add_edge(n, c)
                
#     pos = graphviz_layout(G, prog='dot')
    
#     nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 3, 4], node_color='#E5E5E5', node_size=500, alpha=1.0)
#     nx.draw_networkx_edges(G, pos, width=1.0)
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color='#348ABD')
#     nx.draw_networkx_edge_labels(G, pos, font_size=10, edge_labels=edge_weights, font_color='#E24A33')
            
#     plt.axis("off")     
#     plt.savefig('fixed_aoa_example_graph.png', bbox_inches='tight')
    
# # fixed_AoA_example_graph()
    
# # =============================================================================
# # Find actual expected critical path length.
# # =============================================================================
    
# def expected_critical_path_MC(nc=4, ng=2, samples=1000):    

#     G = nx.DiGraph()    
#     info = {1 : [2, 3], 2 : [3, 4], 3 : [4]}
#     edge_weights = {(1, 2) : (1, 1, 2, 7, 8, 8),
#                     (1, 3) : (1, 2, 3, 7, 8, 8),
#                     (2, 3) : (2, 2, 3, 5, 7, 7),
#                     (2, 4) : (5, 6, 6, 8, 10, 10),
#                     (3, 4) : (4, 5, 5, 7, 9, 8)}
#     for n, kids in info.items():
#         for c in kids:
#             G.add_edge(n, c)
    
#     d = (nc + ng)**2
#     edge_probs = [nc/d, (nc * (nc - 1))/d, (nc * ng) / d,
#                   ng/d, (nc * ng)/ d, (ng * (ng - 1))/d] 
    
#     path_lengths = []
#     for s in range(samples):        
#         for e in G.edges():
#             G.edges[e]['weight'] = np.random.choice(edge_weights[e], p=edge_probs)        
#         length = 0.0
#         # Find the critical path.
#         cp = nx.algorithms.dag.dag_longest_path(G)
#         for i, n in enumerate(cp):
#             if i == 0:
#                 continue
#             e = (cp[i - 1], n)
#             length += G.edges[e]['weight']
#         path_lengths.append(length)
#     return np.mean(path_lengths)

# xcp = expected_critical_path_MC()
# print("Expected critical path length: ~{}".format(xcp))
  


    