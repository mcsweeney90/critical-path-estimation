#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEFT vs Fulkerson.
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

# Variables etc used throughout.

chol_dag_path = '../DAGs/cholesky/'
chol_results_path = 'results/fulkerson/cholesky'
pathlib.Path(chol_results_path).mkdir(parents=True, exist_ok=True)
stg_dag_size = 100
stg_dag_path = '../graphs/STG/{}'.format(stg_dag_size)
stg_results_path = 'results/fulkerson/stg'
pathlib.Path(stg_results_path).mkdir(parents=True, exist_ok=True)

####################################################################################################

start = timer()

n_workers = [2, 4, 8]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]

# reductions = {}
# for platform in platforms:
#     reductions[platform.name] = {}
#     for alpha in alphas:
#         reductions[platform.name][alpha] = {}
#         for beta in betas:
#             reductions[platform.name][alpha][beta] = {}
#             for rk in rankings:
#                 reductions[platform.name][alpha][beta][rk] = []

slrs, speedups = {}, {}
for nw in n_workers:
    slrs[nw], speedups[nw] = {}, {}
    for ccr in ccrs:
        slrs[nw][ccr], speedups[nw][ccr] = {}, {}
        for h in het_factors:
            slrs[nw][ccr][h], speedups[nw][ccr][h] = {}, {}
            slrs[nw][ccr][h]["HEFT"] = []
            slrs[nw][ccr][h]["Fulkerson"] = []
            speedups[nw][ccr][h]["HEFT"] = []
            speedups[nw][ccr][h]["Fulkerson"] = []

for nw in n_workers:
    print("\nStarting {} workers...".format(nw))
    platform = Platform(nw, name="{}P".format(nw)) 
    for ccr in ccrs:
        print("Starting CCR = {}...".format(ccr))
        for het_factor in het_factors:
            print("Starting het factor = {}...".format(het_factor))
            with open("{}/{}_CCR{}_H{}_{}tasks.txt".format(stg_results_path, platform.name, ccr, het_factor, stg_dag_size), "w") as dest:
                print("COMPARISON OF HEFT WITH STANDARD UPWARD RANKING AND FULKERSON-BASED RANKING.", file=dest)
                print("180 DAGs FROM THE STG WITH {} TASKS.".format(stg_dag_size + 2), file=dest)
                print("TARGET DAG COMPUTATION-TO-COMMUNICATION RATIO (CCR): {}. (ACTUAL VALUE MAY DIFFER SLIGHTLY.)".format(ccr), file=dest) 
                print("HETEROGENEITY FACTOR: {}.".format(het_factor), file=dest)                 
                platform.print_info(filepath=dest)
                
                # Iterate over DAG directory.
                reductions = []
                count = 0
                for name in os.listdir('{}'.format(stg_dag_path)):
                    print("Starting {}...".format(name))
                    count += 1
                    # if count > 3:
                    #     break
                    # Load DAG topology.
                    with open('{}/{}'.format(stg_dag_path, name), 'rb') as file:
                        dag = dill.load(file)
                    # Set DAG costs.
                    dag.set_costs(platform, target_ccr=ccr, method="HEFT", het_factor=het_factor)
                    # Print DAG info to file.
                    mst, cp = dag.print_info(return_mst_and_cp=True, filepath=dest) 
                    
                    for h in ["HEFT", "Fulkerson"]:                    
                        mkspan = HEFT(dag, platform, cp_type=h)
                        print("\n{} makespan: {}".format(h, mkspan), file=dest)
                        slr = mkspan / cp
                        slrs[nw][ccr][het_factor][h].append(slr)
                        print("SLR: {}".format(slr), file=dest)
                        speedup = mst / mkspan 
                        speedups[nw][ccr][het_factor][h].append(speedup)
                        print("Speedup: {}".format(speedup), file=dest)
                        if h == "HEFT":
                            m = mkspan
                        else:
                            r = 100 - (mkspan/m)*100
                            reductions.append(r)
                            print("Reduction vs standard HEFT (%) : {}".format(r), file=dest)                                                                  
                    
                    print("--------------------------------------------------------\n", file=dest) 
                print("\n\n\n\n\n", file=dest)
                print("--------------------------------------------------------------------------------", file=dest)
                print("--------------------------------------------------------------------------------", file=dest)
                print("FINAL SUMMARY", file=dest)
                print("--------------------------------------------------------------------------------", file=dest)
                print("--------------------------------------------------------------------------------\n", file=dest)
                print("DAGs considered: {}".format(count), file=dest)
                for h in ["HEFT", "Fulkerson"]:
                    print("\n\nHEURISTIC: {}".format(h), file=dest)
                    
                    print("Mean SLR: {}".format(np.mean(slrs[nw][ccr][het_factor][h])), file=dest)
                    print("Best SLR: {}".format(max(slrs[nw][ccr][het_factor][h])), file=dest)
                    print("Worst SLR: {}".format(min(slrs[nw][ccr][het_factor][h])), file=dest)
                    
                    print("\nMean Speedup: {}".format(np.mean(speedups[nw][ccr][het_factor][h])), file=dest)
                    print("Best Speedup: {}".format(max(speedups[nw][ccr][het_factor][h])), file=dest)
                    print("Worst Speedup: {}".format(min(speedups[nw][ccr][het_factor][h])), file=dest)
                    
                    if h != "HEFT":
                        print("\nMean reduction (%): {}".format(np.mean(reductions)), file=dest)
                        print("Best reduction (%): {}".format(max(reductions)), file=dest)
                        print("Worst reduction (%): {}".format(min(reductions)), file=dest)
                        print("Number of times better: {}/{}".format(sum(1 for r in reductions if r > 0.0), count), file=dest)
                        print("Number of times worse: {}/{}".format(sum(1 for r in reductions if r < 0.0), count), file=dest)
                        
                
# Save the reductions.
with open('{}/slrs.dill'.format(stg_results_path), 'wb') as handle:
    dill.dump(slrs, handle)
with open('{}/speedups.dill'.format(stg_results_path), 'wb') as handle:
    dill.dump(speedups, handle)

elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))
