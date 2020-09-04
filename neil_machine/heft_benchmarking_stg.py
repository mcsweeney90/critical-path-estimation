#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking HEFT on Neil's machine.
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import itertools as it
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Platform, HEFT

results_path = 'results/heft_benchmarking/stg'
pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

sizes = [100]
n_workers = [2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [0.5, 1.0, 1.5, 2.0]
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

with open("{}/execution_info.txt".format(results_path), "w") as dest:
    start = timer()
    for sz in sizes: 
        print("\nStarting size {} graphs...".format(sz), file=dest)
        dag_path = '../graphs/STG/{}'.format(sz)            
        for nw in n_workers:
            print("\nStarting {} workers...".format(nw), file=dest)
            platform = Platform(nw, name="{}P".format(nw)) 
            p_start = timer()
            for ccr in ccrs:
                print("Starting CCR = {}...".format(ccr), file=dest)
                for h in het_factors:
                    print("Starting h = {}...".format(h), file=dest)
                    for m in methods:    
                        print("Starting m = {}...".format(m), file=dest)
                        # Iterate over DAG directory.
                        for name in os.listdir('{}'.format(dag_path)):
                            # Load DAG topology.
                            with open('{}/{}'.format(dag_path, name), 'rb') as file:
                                dag = dill.load(file)                        
                            for _ in range(10):                            
                                # Set DAG costs.
                                dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)
                                mst = dag.minimal_serial_time()
                                OCP = dag.conditional_critical_paths(direction="downward", cp_type="LB")
                                cp = max(min(OCP[task.ID][p] for p in OCP[task.ID]) for task in dag.graph if task.exit)                                 
                                # Find HEFT makespan.
                                heft_mkspan = HEFT(dag, platform)
                                info[sz][nw][ccr][h][m]["makespans"].append(heft_mkspan)
                                # SLR.
                                slr = heft_mkspan / cp
                                info[sz][nw][ccr][h][m]["SLRs"].append(slr)
                                # Speedup.
                                speedup = mst / heft_mkspan
                                info[sz][nw][ccr][h][m]["speedups"].append(speedup)
                                # Improvement compared to random topological sort.
                                rand_mkspan = HEFT(dag, platform, priority_list=dag.top_sort)
                                r = 100 - (heft_mkspan/rand_mkspan)*100
                                info[sz][nw][ccr][h][m]["reductions"].append(r)
            p_elapsed = timer() - p_start
            print("{} workers took {} minutes".format(nw, p_elapsed/60), file=dest)
    elapsed = timer() - start
    print("\nTOTAL TIME: {} minutes".format(elapsed / 60), file=dest)                                           
                
# Save the info.
with open('{}/info.dill'.format(results_path), 'wb') as handle:
    dill.dump(info, handle)
