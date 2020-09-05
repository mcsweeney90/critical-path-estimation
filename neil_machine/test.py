#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing...
"""

import dill
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Task, DAG, Platform, HEFT, PEFT, HSM


# with open('results/sz100_info.dill', 'rb') as file:
#     info = dill.load(file) 

# n_workers = [2, 4, 8, 16]
# ccrs = [0.1, 1, 10]
# het_factors = [1.0, 2.0]

# for nw in n_workers:
#     for ccr in ccrs:
#         for h in het_factors:
#             print("\n nw = {}, ccr = {}, h = {}".format(nw, ccr, h))
#             print(info["rand0001"][nw][ccr][h])


sz = 1000
dag_path = '../graphs/STG/{}'.format(sz)
ccr, h, m = 10, 1.0, "R"
cps = ["LB", "W", "F", "WF"]
ccps = ["LB", "M", "WM"]
with open('{}/rand0085.dill'.format(dag_path), 'rb') as file:
    dag = dill.load(file)      
    
with open("test.txt", "w") as dest: 
    print("SIZE : {}".format(sz), file=dest)
    for nw in [2, 4, 8, 16]:
        start = timer()
        platform = Platform(nw, name="{}P".format(nw)) 
        print("\n{} PROCESSORS".format(nw), file=dest)
        for _ in range(5):
            dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)
            mst = dag.minimal_serial_time()
            OCP = dag.conditional_critical_paths(direction="downward", cp_type="LB")
            cp = max(min(OCP[task.ID][p] for p in OCP[task.ID]) for task in dag.graph if task.exit)                                 
            # Find HEFT makespan.
            heft_mkspan = HEFT(dag, platform)
            # Improvement compared to random topological sort.
            rand_mkspan = HEFT(dag, platform, priority_list=dag.top_sort)
            # Alternative task rankings.
            for cp in cps:
                if cp == "F" or cp == "WF":
                    continue
                avg_type = "WM" if cp == "W" else "HEFT" 
                try:
                    mkspan = HEFT(dag, platform, cp_type=cp, avg_type=avg_type)  
                except KeyError:
                    pass 
            # PEFT.
            peft_mkspan = PEFT(dag, platform) 
            # Alternative processor selections.
            for ccp in ccps: 
                CCP = dag.conditional_critical_paths(cp_type=ccp, lookahead=True)
                priority_list = dag.conditional_critical_path_priorities(platform, CCP=CCP, cp_type=ccp)
                mkspan = HSM(dag, platform, cp_type=ccp, CCP=CCP, priority_list=priority_list) 
                mks = HEFT(dag, platform, priority_list=priority_list)
        elapsed = timer() - start
        print("Time for 5 runs with single DAG: {} mins".format(elapsed / 60), file=dest)
        est_time = elapsed * 180 * 3 * 2 
        print("Estimated time for entire set (180 DAGs, 2 x h, 3 x CCR): {} mins".format(est_time / 60), file=dest)
