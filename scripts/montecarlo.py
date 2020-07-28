#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo.
Does "improving" the critical path estimate
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
chol_results_path = 'results/MC/cholesky'
pathlib.Path(chol_results_path).mkdir(parents=True, exist_ok=True)
stg_dag_size = 100
stg_dag_path = '../graphs/STG/{}'.format(stg_dag_size)
stg_results_path = 'results/MC/stg'
pathlib.Path(stg_results_path).mkdir(parents=True, exist_ok=True)

####################################################################################################

start = timer()

n_workers = [2, 4, 8]
ccrs = [0.1, 1, 10]
het_factors = [2.0]
nsamples = [1, 10, 100, 1000]

reductions = {}
for nw in n_workers:
    reductions[nw] = {}
    for ccr in ccrs:
        reductions[nw][ccr] = {}
        for h in het_factors:
            reductions[nw][ccr][h] = {}
            for s in nsamples:
                reductions[nw][ccr][h]["MC{}".format(s)] = []    
                reductions[nw][ccr][h]["WMC{}".format(s)] = []

for nw in n_workers:
    print("\nStarting {} workers...".format(nw))
    platform = Platform(nw, name="{}P".format(nw)) 
    for ccr in ccrs:
        print("Starting CCR = {}...".format(ccr))
        for h in het_factors:
            print("Starting het factor = {}...".format(h))
            with open("{}/{}_CCR{}_H{}_{}tasks.txt".format(stg_results_path, platform.name, ccr, h, stg_dag_size), "w") as dest:
                print("COMPARISON OF HEFT WITH STANDARD UPWARD RANKING AND MONTE CARLO-BASED RANKINGS.", file=dest)
                print("180 DAGs FROM THE STG WITH {} TASKS.".format(stg_dag_size + 2), file=dest)
                print("TARGET DAG COMPUTATION-TO-COMMUNICATION RATIO (CCR): {}. (ACTUAL VALUE MAY DIFFER SLIGHTLY.)".format(ccr), file=dest) 
                print("HETEROGENEITY FACTOR: {}.".format(h), file=dest)                 
                platform.print_info(filepath=dest)
                
                failures = {}
                for ns in nsamples:
                    failures["MC{}".format(ns)] = 0
                    failures["WMC{}".format(ns)] = 0
                failures["HEFT"] = 0
                # Iterate over DAG directory.
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
                    dag.set_costs(platform, target_ccr=ccr, method="related", het_factor=h)
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
                    
                    for ns in nsamples:
                        for cp in ["MC", "WMC"]:
                            p_list = dag.critical_path_priorities(cp_type=cp, mc_samples=ns)
                            mkspan = HEFT(dag, platform, priority_list=p_list)
                            if mkspan > mst:
                                failures["{}{}".format(cp, ns)] += 1
                            print("\nHEFT-{}{} makespan: {}".format(cp, ns, mkspan), file=dest)
                            r = 100 - (mkspan/heft_mkspan)*100
                            reductions[nw][ccr][h]["{}{}".format(cp, ns)].append(r)
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
                for ns in nsamples:
                    for t in ["MC", "WMC"]:
                        cp = t + str(ns)
                        print("\nHEURISTIC: HEFT-{}".format(cp), file=dest)    
                        print("Number of failures: {}".format(failures[cp]), file=dest)
                        print("Mean reduction (%): {}".format(np.mean(reductions[nw][ccr][h][cp])), file=dest)
                        print("Best reduction (%): {}".format(max(reductions[nw][ccr][h][cp])), file=dest)
                        print("Worst reduction (%): {}".format(min(reductions[nw][ccr][h][cp])), file=dest)
                        print("Number of times better: {}/{}".format(sum(1 for r in reductions[nw][ccr][h][cp] if r > 0.0), count), file=dest)
                        print("Number of times worse: {}/{}".format(sum(1 for r in reductions[nw][ccr][h][cp] if r < 0.0), count), file=dest)
                        
                
# Save the reductions.
with open('{}/reductions.dill'.format(stg_results_path), 'wb') as handle:
    dill.dump(reductions, handle)

elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))