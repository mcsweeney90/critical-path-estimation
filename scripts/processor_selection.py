#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processor selection.
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Platform, HEFT, PEFT, HSM

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
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.

####################################################################################################

# =============================================================================
# Testing.
# =============================================================================

nw = 2
platform = Platform(nw, name="{}P".format(nw))
sz, ccr, m, h = 100, 10, "R", 2.0
dag_path = '../graphs/STG/{}'.format(sz)
with open('{}/rand0000.dill'.format(dag_path), 'rb') as file:
    dag = dill.load(file)
dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)

mst = dag.minimal_serial_time()
print("MST = {}".format(mst))
heft_mkspan = HEFT(dag, platform)
print("HEFT makespan: {}".format(heft_mkspan))
peft_mkspan = PEFT(dag, platform)
print("PEFT makespan: {}".format(peft_mkspan))
hsm_mkspan = HSM(dag, platform, rank_avg="WM")
print("HSM makespan: {}".format(hsm_mkspan))

# OCT = dag.optimistic_cost_table()
# # CCP = dag.conditional_critical_paths(cp_type="LB", lookahead=True)
# # Compute task ranks.
# task_ranks = {}
# for t in dag.top_sort:
#     s = sum(1/(t.comp_costs[w.ID] + OCT[t.ID][w.ID]) for w in platform.workers)
#     task_ranks[t] = platform.n_workers / s
# priority_list = list(sorted(task_ranks, key=task_ranks.get, reverse=True))

# done = set()
# for t in priority_list:
#     for s in dag.graph.successors(t):
#         if s.ID in done:
#             print("Parent: {}".format(t.ID))
#             print("Parent rank: {}".format(task_ranks[t]))
#             print("Child: {}".format(s.ID))
#             print("Child rank: {}".format(task_ranks[s]))
#             break
#     done.add(t.ID)
            

# =============================================================================
# Size 100 DAGs from the STG.
# =============================================================================

# start = timer()

# sz = 100
# dag_path = '../graphs/STG/{}'.format(sz)
# results_path = 'results/processor_selection/stg/{}'.format(sz)
# pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

# n_workers = [2, 4, 8]
# ccrs = [0.1, 1, 10]
# het_factors = [1.0, 2.0]
# methods = ["R", "UR"]

# rks = ["LB", "W"]
# makespans = {}
# for nw in n_workers:
#     makespans[nw] = {}
#     for ccr in ccrs:
#         makespans[nw][ccr] = {}
#         for h in het_factors:
#             makespans[nw][ccr][h] = {}
#             for m in methods:
#                 makespans[nw][ccr][h][m] = {}
#                 makespans[nw][ccr][h][m]["HEFT"] = []
#                 makespans[nw][ccr][h][m]["MST"] = []
#                 makespans[nw][ccr][h][m]["CP"] = []                
#                 for rk in rks:
#                     makespans[nw][ccr][h][m][rk] = []                

# for nw in n_workers:
#     print("\nStarting {} workers...".format(nw))
#     platform = Platform(nw, name="{}P".format(nw)) 
#     for ccr in ccrs:
#         print("Starting CCR = {}...".format(ccr))
#         for h in het_factors:
#             print("Starting het factor = {}...".format(h))
#             for m in methods:
#                 print("Starting method {}...".format(m))
#                 with open("{}/{}_CCR{}_H{}_{}.txt".format(results_path, platform.name, ccr, h, m), "w") as dest:
#                     print("COMPARISON OF HEFT WITH STANDARD UPWARD RANKING AND ALTERNATIVE RANKINGS.", file=dest)
#                     print("TARGET PLATFORM: {} PROCESSORS.".format(nw), file=dest)
#                     print("180 DAGs FROM THE STG WITH {} TASKS.".format(sz + 2), file=dest)
#                     print("TARGET DAG COMPUTATION-TO-COMMUNICATION RATIO (CCR): {}. (ACTUAL VALUE MAY DIFFER SLIGHTLY.)".format(ccr), file=dest) 
#                     print("HETEROGENEITY FACTOR: {}.".format(h), file=dest) 
#                     print("COST GENERATION METHOD: {}.".format(m), file=dest)
#                     platform.print_info(filepath=dest)
                    
#                     reductions = {rk : [] for rk in rks}
#                     failures = {rk : 0 for rk in rks}
#                     failures["HEFT"] = 0
                    
#                     # Iterate over DAG directory.
#                     count = 0
#                     for name in os.listdir('{}'.format(dag_path)):
#                         print("Starting {}...".format(name))
#                         count += 1
#                         # if count > 3:
#                         #     break
#                         # Load DAG topology.
#                         with open('{}/{}'.format(dag_path, name), 'rb') as file:
#                             dag = dill.load(file)
#                         # Set DAG costs.
#                         dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)
#                         # Print DAG info to file.
#                         mst, cp = dag.print_info(return_mst_and_cp=True, filepath=dest)
#                         makespans[nw][ccr][h][m]["MST"].append(mst)
#                         makespans[nw][ccr][h][m]["CP"].append(cp)
                        
#                         # Find HEFT makespan.
#                         heft_mkspan = HEFT(dag, platform)
#                         print("\nHEFT makespan: {}".format(heft_mkspan), file=dest)
#                         makespans[nw][ccr][h][m]["HEFT"].append(heft_mkspan)
#                         slr = heft_mkspan / cp
#                         print("SLR: {}".format(slr), file=dest)
#                         speedup = mst / heft_mkspan
#                         print("Speedup: {}".format(speedup), file=dest)
#                         if speedup < 1.0:
#                             failures["HEFT"] += 1
                        
#                         for rk in rks:
#                             avg_type = "WM" if rk == "W" else "HEFT" 
#                             mkspan = HEFT(dag, platform, cp_type=rk, avg_type=avg_type) 
#                             print("\nHEFT-{} makespan: {}".format(rk, mkspan), file=dest)
#                             if mkspan > mst:
#                                 failures[rk] += 1
#                             makespans[nw][ccr][h][m][rk].append(mkspan)
#                             r = 100 - (mkspan/heft_mkspan)*100
#                             reductions[rk].append(r)
#                             print("Reduction vs standard HEFT (%) : {}".format(r), file=dest)                                                            
                        
#                         print("--------------------------------------------------------\n", file=dest) 
#                     print("\n\n\n\n\n", file=dest)
#                     print("--------------------------------------------------------------------------------", file=dest)
#                     print("--------------------------------------------------------------------------------", file=dest)
#                     print("FINAL SUMMARY", file=dest)
#                     print("--------------------------------------------------------------------------------", file=dest)
#                     print("--------------------------------------------------------------------------------\n", file=dest)
#                     print("DAGs considered: {}".format(count), file=dest)
#                     print("Number of HEFT failures: {}".format(failures["HEFT"]), file=dest)
#                     for rk in rks:
#                         print("\nRANKING: {}".format(rk), file=dest) 
#                         print("Number of failures: {}".format(failures[rk]), file=dest)
#                         print("Mean reduction (%): {}".format(np.mean(reductions[rk])), file=dest)
#                         print("Best reduction (%): {}".format(max(reductions[rk])), file=dest)
#                         print("Worst reduction (%): {}".format(min(reductions[rk])), file=dest)
#                         print("Number of times better: {}/{}".format(sum(1 for r in reductions[rk] if r > 0.0), count), file=dest)
#                         print("Number of times worse: {}/{}".format(sum(1 for r in reductions[rk] if r < 0.0), count), file=dest)                        
                
# # Save the reductions.
# with open('{}/makespans.dill'.format(results_path), 'wb') as handle:
#     dill.dump(makespans, handle)

# elapsed = timer() - start
# print("This took {} minutes".format(elapsed / 60))