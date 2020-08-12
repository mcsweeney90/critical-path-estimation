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

start = timer()

sz = 100
dag_path = '../graphs/STG/{}'.format(sz)
results_path = 'results/fulkerson/stg/{}'.format(sz)
pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

n_workers = [2, 4, 8]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
methods = ["R", "UR"]

cps = ["LB", "W", "F", "WF"]
reductions = {}
for nw in n_workers:
    reductions[nw] = {}
    for ccr in ccrs:
        reductions[nw][ccr] = {}
        for h in het_factors:
            reductions[nw][ccr][h] = {}
            for m in methods:
                reductions[nw][ccr][h][m] = {}
                for cp in cps:
                    reductions[nw][ccr][h][m][cp] = []                

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
                    
                    failures = {cp : 0 for cp in cps}
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
                        
                        # Find HEFT makespan.
                        heft_mkspan = HEFT(dag, platform)
                        print("\nHEFT makespan: {}".format(heft_mkspan), file=dest)
                        slr = heft_mkspan / cp
                        print("SLR: {}".format(slr), file=dest)
                        speedup = mst / heft_mkspan
                        print("Speedup: {}".format(speedup), file=dest)
                        if speedup < 1.0:
                            failures["HEFT"] += 1
                        
                        for cp in cps:
                            avg_type = "WM" if cp == "W" else "HEFT" 
                            try:
                                mkspan = HEFT(dag, platform, cp_type=cp, avg_type=avg_type) 
                                print("\nHEFT-{} makespan: {}".format(cp, mkspan), file=dest)
                                if mkspan > mst:
                                    failures[cp] += 1
                                r = 100 - (mkspan/heft_mkspan)*100
                                reductions[nw][ccr][h][m][cp].append(r)
                                print("Reduction vs standard HEFT (%) : {}".format(r), file=dest)  
                            except KeyError:
                                print("\nHEFT-{} FAILED".format(cp), file=dest)                                                             
                        
                        print("--------------------------------------------------------\n", file=dest) 
                    print("\n\n\n\n\n", file=dest)
                    print("--------------------------------------------------------------------------------", file=dest)
                    print("--------------------------------------------------------------------------------", file=dest)
                    print("FINAL SUMMARY", file=dest)
                    print("--------------------------------------------------------------------------------", file=dest)
                    print("--------------------------------------------------------------------------------\n", file=dest)
                    print("DAGs considered: {}".format(count), file=dest)
                    print("Number of HEFT failures: {}".format(failures["HEFT"]), file=dest)
                    for cp in cps:
                        print("\nRANKING: {}".format(cp), file=dest) 
                        print("Number of failures: {}".format(failures[cp]), file=dest)
                        print("Mean reduction (%): {}".format(np.mean(reductions[nw][ccr][h][m][cp])), file=dest)
                        print("Best reduction (%): {}".format(max(reductions[nw][ccr][h][m][cp])), file=dest)
                        print("Worst reduction (%): {}".format(min(reductions[nw][ccr][h][m][cp])), file=dest)
                        print("Number of times better: {}/{}".format(sum(1 for r in reductions[nw][ccr][h][m][cp] if r > 0.0), count), file=dest)
                        print("Number of times worse: {}/{}".format(sum(1 for r in reductions[nw][ccr][h][m][cp] if r < 0.0), count), file=dest)                        
                
# Save the reductions.
with open('{}/reductions.dill'.format(results_path), 'wb') as handle:
    dill.dump(reductions, handle)

elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))
