#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEFT with alternative critical path rankings.
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Platform, HEFT

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
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.

####################################################################################################

# =============================================================================
# First, size 100 graphs (since Fulkerson too slow for 1000).
# =============================================================================

# start = timer()

# sz = 100
# dag_path = '../graphs/STG/{}'.format(sz)
# results_path = 'results/fulkerson/stg/{}'.format(sz)
# pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

# n_workers = [2, 4, 8]
# ccrs = [0.1, 1, 10]
# het_factors = [1.0, 2.0]
# methods = ["R", "UR"]

# cps = ["LB", "W", "F", "WF"]
# reductions = {}
# for nw in n_workers:
#     reductions[nw] = {}
#     for ccr in ccrs:
#         reductions[nw][ccr] = {}
#         for h in het_factors:
#             reductions[nw][ccr][h] = {}
#             for m in methods:
#                 reductions[nw][ccr][h][m] = {}
#                 for cp in cps:
#                     reductions[nw][ccr][h][m][cp] = []                

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
                    
#                     failures = {cp : 0 for cp in cps}
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
                        
#                         # Find HEFT makespan.
#                         heft_mkspan = HEFT(dag, platform)
#                         print("\nHEFT makespan: {}".format(heft_mkspan), file=dest)
#                         slr = heft_mkspan / cp
#                         print("SLR: {}".format(slr), file=dest)
#                         speedup = mst / heft_mkspan
#                         print("Speedup: {}".format(speedup), file=dest)
#                         if speedup < 1.0:
#                             failures["HEFT"] += 1
                        
#                         for cp in cps:
#                             avg_type = "WM" if cp == "W" else "HEFT" 
#                             try:
#                                 mkspan = HEFT(dag, platform, cp_type=cp, avg_type=avg_type) 
#                                 print("\nHEFT-{} makespan: {}".format(cp, mkspan), file=dest)
#                                 if mkspan > mst:
#                                     failures[cp] += 1
#                                 r = 100 - (mkspan/heft_mkspan)*100
#                                 reductions[nw][ccr][h][m][cp].append(r)
#                                 print("Reduction vs standard HEFT (%) : {}".format(r), file=dest)  
#                             except KeyError:
#                                 print("\nHEFT-{} FAILED".format(cp), file=dest)                                                             
                        
#                         print("--------------------------------------------------------\n", file=dest) 
#                     print("\n\n\n\n\n", file=dest)
#                     print("--------------------------------------------------------------------------------", file=dest)
#                     print("--------------------------------------------------------------------------------", file=dest)
#                     print("FINAL SUMMARY", file=dest)
#                     print("--------------------------------------------------------------------------------", file=dest)
#                     print("--------------------------------------------------------------------------------\n", file=dest)
#                     print("DAGs considered: {}".format(count), file=dest)
#                     print("Number of HEFT failures: {}".format(failures["HEFT"]), file=dest)
#                     for cp in cps:
#                         print("\nRANKING: {}".format(cp), file=dest) 
#                         print("Number of failures: {}".format(failures[cp]), file=dest)
#                         print("Mean reduction (%): {}".format(np.mean(reductions[nw][ccr][h][m][cp])), file=dest)
#                         print("Best reduction (%): {}".format(max(reductions[nw][ccr][h][m][cp])), file=dest)
#                         print("Worst reduction (%): {}".format(min(reductions[nw][ccr][h][m][cp])), file=dest)
#                         print("Number of times better: {}/{}".format(sum(1 for r in reductions[nw][ccr][h][m][cp] if r > 0.0), count), file=dest)
#                         print("Number of times worse: {}/{}".format(sum(1 for r in reductions[nw][ccr][h][m][cp] if r < 0.0), count), file=dest)                        
                
# # Save the reductions.
# with open('{}/reductions.dill'.format(results_path), 'wb') as handle:
#     dill.dump(reductions, handle)

# elapsed = timer() - start
# print("This took {} minutes".format(elapsed / 60))

# =============================================================================
# Plots.
# =============================================================================

# TODO: What's the best way to visualize this data? 
# sz = 100
# results_path = 'results/fulkerson/stg/{}'.format(sz)
# with open('{}/reductions.dill'.format(results_path), 'rb') as file:
#     reductions = dill.load(file) 
    
# n_workers = [2, 4, 8]
# ccrs = [0.1, 1, 10]
# het_factors = [1.0, 2.0]
# methods = ["R", "UR"]

# length, width = 3, 0.16
# x = np.arange(length)
# xlabels = [0.1, 1.0, 10]

# cps = ["LB", "W", "F", "WF"]
# colors = {"LB" : '#E24A33', "W" : '#348ABD', "F" : '#FBC15E', "WF" : '#8EBA42'}

# fig = plt.figure(dpi=400) 
# ax = fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axes
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# ax.set_xlabel("COMPUTATION-TO-COMMUNICATION RATIO", labelpad=5)
# ax.set_ylabel("BETTER - WORSE (%)", labelpad=10)

# for i, nw in enumerate(n_workers):
#     ax1 = fig.add_subplot(311 + i)
    
#     differential = {cp : [0.0, 0.0, 0.0] for cp in cps}
#     for h, m in [(1.0, "R"), (1.0, "UR"), (2.0, "R"), (2.0, "UR")]:
#           for cp in cps:
#               for k, ccr in enumerate(ccrs):
#                   b = sum(1 for r in reductions[nw][ccr][h][m][cp] if r > 0.0)
#                   w = sum(1 for r in reductions[nw][ccr][h][m][cp] if r < 0.0)
#                   p = ((b - w)/180) * 25 
#                   differential[cp][k] += p
#     # avg_reductions = {cp : [0, 0, 0] for cp in cps}
#     # for h, m in [(1.0, "R"), (1.0, "UR"), (2.0, "R"), (2.0, "UR")]:
#     #      for cp in cps:
#     #          for k, ccr in enumerate(ccrs):
#     #              r = np.mean(reductions[nw][ccr][h][m][cp])/4
#     #              avg_reductions[cp][k] += r    
    
#     j = 0
#     for cp in cps:
#         ax1.bar(x + j * width, differential[cp], width, color=colors[cp], edgecolor='white', label=cp)
#         j += 1   
#     ax1.set_xticks(x + (3/2) * width)
#     if i < 2:
#         ax1.set_xticklabels([]) 
#         if i == 0:
#             ax1.legend(handlelength=3, handletextpad=0.4, ncol=4, loc='best', fancybox=True, facecolor='white')                
#     else:
#         ax1.set_xticklabels(xlabels)
#     # plt.axhline(y=50,linewidth=1, color='k')
#     ax1.set_title("{} PROCESSORS".format(nw), color="black", family='serif')
    
# plt.savefig('{}/{}tasks_differential'.format(results_path, sz), bbox_inches='tight') 
# plt.close(fig) 

# =============================================================================
# Print average reductions (for table in write up).
# =============================================================================

# sz = 100
# results_path = 'results/fulkerson/stg/{}'.format(sz)
# with open('{}/reductions.dill'.format(results_path), 'rb') as file:
#     reductions = dill.load(file)

# cps = ["LB", "W", "F", "WF"]
# n_workers = [2, 4, 8]
# ccrs = [0.1, 1, 10]

# avg_reductions = {}
# for nw in n_workers:
#     avg_reductions[nw] = {cp : [0, 0, 0] for cp in cps}    

# for nw in n_workers:  
#     for h, m in [(1.0, "R"), (1.0, "UR"), (2.0, "R"), (2.0, "UR")]:
#           for cp in cps:
#               for k, ccr in enumerate(ccrs):
#                   r = np.mean(reductions[nw][ccr][h][m][cp])/4
#                   avg_reductions[nw][cp][k] += r  

# # Print to file.
# with open("{}/avg_reductions.txt".format(results_path), "w") as dest:
#     for nw in n_workers:
#         print("\n{} PROCESSORS".format(nw), file=dest)
#         for cp in cps:
#             print("Ranking: {}".format(cp), file=dest)
#             for k, ccr in enumerate(ccrs):
#                 print("CCR {} : {}".format(ccr, avg_reductions[nw][cp][k]), file=dest)

# =============================================================================
# Size 1000 DAGs from the STG.
# =============================================================================

start = timer()

sz = 1000
dag_path = '../graphs/STG/{}'.format(sz)
results_path = 'results/fulkerson/stg/{}'.format(sz)
pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

n_workers = [2, 4, 8]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
methods = ["R", "UR"]

rks = ["LB", "W"]
makespans = {}
for nw in n_workers:
    makespans[nw] = {}
    for ccr in ccrs:
        makespans[nw][ccr] = {}
        for h in het_factors:
            makespans[nw][ccr][h] = {}
            for m in methods:
                makespans[nw][ccr][h][m] = {}
                makespans[nw][ccr][h][m]["HEFT"] = []
                makespans[nw][ccr][h][m]["MST"] = []
                makespans[nw][ccr][h][m]["CP"] = []                
                for rk in rks:
                    makespans[nw][ccr][h][m][rk] = []                

for nw in n_workers:
    print("\nStarting {} workers...".format(nw))
    platform = Platform(nw, name="{}P".format(nw)) 
    for ccr in ccrs:
        print("Starting CCR = {}...".format(ccr))
        for h in het_factors:
            print("Starting het factor = {}...".format(h))
            for m in methods:
                print("Starting method {}...".format(m))
                with open("{}/{}_CCR{}_H{}_{}.txt".format(results_path, platform.name, ccr, h, m), "w") as dest:
                    print("COMPARISON OF HEFT WITH STANDARD UPWARD RANKING AND ALTERNATIVE RANKINGS.", file=dest)
                    print("TARGET PLATFORM: {} PROCESSORS.".format(nw), file=dest)
                    print("180 DAGs FROM THE STG WITH {} TASKS.".format(sz + 2), file=dest)
                    print("TARGET DAG COMPUTATION-TO-COMMUNICATION RATIO (CCR): {}. (ACTUAL VALUE MAY DIFFER SLIGHTLY.)".format(ccr), file=dest) 
                    print("HETEROGENEITY FACTOR: {}.".format(h), file=dest) 
                    print("COST GENERATION METHOD: {}.".format(m), file=dest)
                    platform.print_info(filepath=dest)
                    
                    reductions = {rk : [] for rk in rks}
                    failures = {rk : 0 for rk in rks}
                    failures["HEFT"] = 0
                    
                    # Iterate over DAG directory.
                    count = 0
                    for name in os.listdir('{}'.format(dag_path)):
                        print("Starting {}...".format(name))
                        count += 1
                        # if count > 3:
                        #     break
                        # Load DAG topology.
                        with open('{}/{}'.format(dag_path, name), 'rb') as file:
                            dag = dill.load(file)
                        # Set DAG costs.
                        dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)
                        # Print DAG info to file.
                        mst, cp = dag.print_info(return_mst_and_cp=True, filepath=dest)
                        makespans[nw][ccr][h][m]["MST"].append(mst)
                        makespans[nw][ccr][h][m]["CP"].append(cp)
                        
                        # Find HEFT makespan.
                        heft_mkspan = HEFT(dag, platform)
                        print("\nHEFT makespan: {}".format(heft_mkspan), file=dest)
                        makespans[nw][ccr][h][m]["HEFT"].append(heft_mkspan)
                        slr = heft_mkspan / cp
                        print("SLR: {}".format(slr), file=dest)
                        speedup = mst / heft_mkspan
                        print("Speedup: {}".format(speedup), file=dest)
                        if speedup < 1.0:
                            failures["HEFT"] += 1
                        
                        for rk in rks:
                            avg_type = "WM" if rk == "W" else "HEFT" 
                            mkspan = HEFT(dag, platform, cp_type=rk, avg_type=avg_type) 
                            print("\nHEFT-{} makespan: {}".format(rk, mkspan), file=dest)
                            if mkspan > mst:
                                failures[rk] += 1
                            makespans[nw][ccr][h][m][rk].append(mkspan)
                            r = 100 - (mkspan/heft_mkspan)*100
                            reductions[rk].append(r)
                            print("Reduction vs standard HEFT (%) : {}".format(r), file=dest)                                                            
                        
                        print("--------------------------------------------------------\n", file=dest) 
                    print("\n\n\n\n\n", file=dest)
                    print("--------------------------------------------------------------------------------", file=dest)
                    print("--------------------------------------------------------------------------------", file=dest)
                    print("FINAL SUMMARY", file=dest)
                    print("--------------------------------------------------------------------------------", file=dest)
                    print("--------------------------------------------------------------------------------\n", file=dest)
                    print("DAGs considered: {}".format(count), file=dest)
                    print("Number of HEFT failures: {}".format(failures["HEFT"]), file=dest)
                    for rk in rks:
                        print("\nRANKING: {}".format(rk), file=dest) 
                        print("Number of failures: {}".format(failures[rk]), file=dest)
                        print("Mean reduction (%): {}".format(np.mean(reductions[rk])), file=dest)
                        print("Best reduction (%): {}".format(max(reductions[rk])), file=dest)
                        print("Worst reduction (%): {}".format(min(reductions[rk])), file=dest)
                        print("Number of times better: {}/{}".format(sum(1 for r in reductions[rk] if r > 0.0), count), file=dest)
                        print("Number of times worse: {}/{}".format(sum(1 for r in reductions[rk] if r < 0.0), count), file=dest)                        
                
# Save the reductions.
with open('{}/makespans.dill'.format(results_path), 'wb') as handle:
    dill.dump(makespans, handle)

elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))
            