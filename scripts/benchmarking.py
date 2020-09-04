#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking.
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
# STG benchmarking.
# =============================================================================

start = timer()

sizes = [100, 1000]
n_workers = [2, 4, 8]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
methods = ["R", "UR"]

info = {}
for sz in sizes:
    info[sz] = {}
    for nw in n_workers:
        info[sz][nw] = {}
        for ccr in ccrs:
            info[sz][nw][ccr] = {}
            for h in het_factors:
                info[sz][nw][ccr][h] = {}
                for m in methods:
                    info[sz][nw][ccr][h][m] = {}                    
                    info[sz][nw][ccr][h][m]["makespans"] = []
                    info[sz][nw][ccr][h][m]["SLRs"] = []
                    info[sz][nw][ccr][h][m]["speedups"] = []
                    info[sz][nw][ccr][h][m]["reductions"] = []

for sz in sizes:   
    print("\nStarting size {} graphs...".format(sz))
    dag_path = '../graphs/STG/{}'.format(sz)
    results_path = 'results/benchmarking/stg/{}'.format(sz)
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)            
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
                        print("BENCHMARKING OF HEFT.", file=dest)
                        print("TARGET PLATFORM: {} PROCESSORS.".format(nw), file=dest)
                        print("180 DAGs FROM THE STG WITH {} TASKS.".format(sz + 2), file=dest)
                        print("TARGET DAG COMPUTATION-TO-COMMUNICATION RATIO (CCR): {}. (ACTUAL VALUE MAY DIFFER SLIGHTLY.)".format(ccr), file=dest) 
                        print("HETEROGENEITY FACTOR: {}.".format(h), file=dest)  
                        print("COST GENERATION METHOD: {}.".format(m), file=dest)
                        platform.print_info(filepath=dest)
                        
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
                            
                            # Find HEFT makespan.
                            heft_mkspan = HEFT(dag, platform)
                            print("\nHEFT makespan: {}".format(heft_mkspan), file=dest)
                            info[sz][nw][ccr][h][m]["makespans"].append(heft_mkspan)
                            slr = heft_mkspan / cp
                            print("SLR: {}".format(slr), file=dest)
                            info[sz][nw][ccr][h][m]["SLRs"].append(slr)
                            speedup = mst / heft_mkspan
                            print("Speedup: {}".format(speedup), file=dest)
                            info[sz][nw][ccr][h][m]["speedups"].append(speedup)
                            # Improvement compared to random topological sort.
                            rand_mkspan = HEFT(dag, platform, priority_list=dag.top_sort)
                            r = 100 - (heft_mkspan/rand_mkspan)*100
                            print("Reduction vs random topological sort: {}".format(r), file=dest)
                            info[sz][nw][ccr][h][m]["reductions"].append(r)
                            print("--------------------------------------------------------\n", file=dest) 
                        print("\n\n\n\n\n", file=dest)
                        print("--------------------------------------------------------------------------------", file=dest)
                        print("--------------------------------------------------------------------------------", file=dest)
                        print("FINAL SUMMARY", file=dest)
                        print("--------------------------------------------------------------------------------", file=dest)
                        print("--------------------------------------------------------------------------------\n", file=dest)
                        print("DAGs considered: {}".format(count), file=dest)
                        print("\nMean SLR: {}".format(np.mean(info[sz][nw][ccr][h][m]["SLRs"])), file=dest)
                        print("Number of times achieved lower bound: {}".format(sum(1 for s in info[sz][nw][ccr][h][m]["SLRs"] if s == 1.0)), file=dest)
                        print("\nMean speedup: {}".format(np.mean(info[sz][nw][ccr][h][m]["speedups"])), file=dest)
                        print("Number of failures: {}".format(sum(1 for s in info[sz][nw][ccr][h][m]["speedups"] if s < 1.0)), file=dest)
                        print("\nNumber of times better than random topological sort: {}/{}".format(sum(1 for r in info[sz][nw][ccr][h][m]["reductions"] if r >= 0.0), count), file=dest)
                        print("Mean reduction vs random topological sort: {}".format(np.mean(info[sz][nw][ccr][h][m]["reductions"])), file=dest)
                        print("Best reduction vs random topological sort: {}".format(max(info[sz][nw][ccr][h][m]["reductions"])), file=dest)
                        print("Worst reduction vs random topological sort: {}".format(min(info[sz][nw][ccr][h][m]["reductions"])), file=dest)
                     
                
# Save the info.
with open('results/benchmarking/stg/info.dill', 'wb') as handle:
    dill.dump(info, handle)

elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))

# =============================================================================
# Plots for STG.
# =============================================================================

with open('results/benchmarking/stg/info.dill', 'rb') as file:
    info = dill.load(file) 
    
sizes = [100, 1000]
n_workers = [2, 4, 8]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
methods = ["R", "UR"]

length, width = 3, 0.16
x = np.arange(length)
xlabels = [0.1, 1.0, 10]

for sz in sizes:
    fig = plt.figure(dpi=400) 
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("CCR", labelpad=5)
    ax.set_ylabel("AVG. REDUCTION (%)", labelpad=10)
    
    for i, nw in enumerate(n_workers):
        ax1 = fig.add_subplot(311 + i)
        
        avg_reductions = {}
        for h, m in [(1.0, "R"), (1.0, "UR"), (2.0, "R"), (2.0, "UR")]:
            avg_reductions[(h, m)] = [np.mean(info[sz][nw][ccr][h][m]["reductions"]) for ccr in ccrs] 
        j = 0
        for h, m in [(1.0, "R"), (1.0, "UR"), (2.0, "R"), (2.0, "UR")]:
            bcolor = '#E24A33' if m == "UR" else '#348ABD'
            bhatch = 'x' if h == 2.0 else None
            blabel = "h = {}, m = {}".format(h, m)
            ax1.bar(x + j * width, avg_reductions[(h, m)], width, color=bcolor, hatch=bhatch, edgecolor='white', label=blabel)
            j += 1   
        ax1.set_xticks(x + (3/2) * width)
        if i < 2:
            ax1.set_xticklabels([]) 
            if i == 0:
                ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white')                
        else:
            ax1.set_xticklabels(xlabels)
        
        ax1.set_title("{} PROCESSORS".format(nw), color="black", family='serif')
        
    plt.savefig('results/benchmarking/stg/{}tasks_reductions'.format(sz), bbox_inches='tight') 
    plt.close(fig) 
