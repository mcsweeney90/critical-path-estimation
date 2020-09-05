#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big script for all results.
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import itertools as it
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Platform, HEFT, PEFT, HSM

sz = 1000
dag_path = '../graphs/STG/{}'.format(sz)
results_path = 'results/'
pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

n_workers = [2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
rks = ["LB", "W", "F", "WF"]
pss = ["LB", "M", "WM"]

# Initialize info dict.
info = {}
for name in os.listdir('{}'.format(dag_path)):
    ns = name[:8]
    info[ns] = {}
    for nw in n_workers:
        info[ns][nw] = {}
        for ccr in ccrs:
            info[ns][nw][ccr] = {}
            for h in het_factors:
                info[ns][nw][ccr][h] = {}
                info[ns][nw][ccr][h]["MST"] = []
                info[ns][nw][ccr][h]["CP"] = []
                info[ns][nw][ccr][h]["HEFT"] = []
                info[ns][nw][ccr][h]["HEFT-R"] = []
                for rk in rks:
                    info[ns][nw][ccr][h]["HEFT-" + rk] = []
                info[ns][nw][ccr][h]["PEFT"] = []
                for ps in pss:
                    info[ns][nw][ccr][h]["PEFT-" + ps] = []
                    info[ns][nw][ccr][h]["PEFT-" + ps + "-NPS"] = []
# Compute.         
with open("{}/sz1000_timing.txt".format(results_path), "w") as dest:
    start = timer()
    for name in os.listdir('{}'.format(dag_path)):
        ns = name[:8]
        # Load DAG topology.
        with open('{}/{}'.format(dag_path, name), 'rb') as file:
            dag = dill.load(file)
        for nw in n_workers:
            platform = Platform(nw, name="{}P".format(nw))
            for ccr in ccrs:
                for h in het_factors:
                    for _ in range(5):
                        # Set DAG costs.
                        dag.set_costs(platform, target_ccr=ccr, method="R", het_factor=h)
                        # Minimal serial time.
                        mst = dag.minimal_serial_time()
                        info[ns][nw][ccr][h]["MST"].append(mst)
                        # Critical path.
                        OCP = dag.conditional_critical_paths(direction="downward", cp_type="LB")
                        cp = max(min(OCP[task.ID][p] for p in OCP[task.ID]) for task in dag.graph if task.exit)
                        info[ns][nw][ccr][h]["CP"].append(cp)
                        # HEFT.
                        mkspan = HEFT(dag, platform)
                        info[ns][nw][ccr][h]["HEFT"].append(mkspan)
                        # HEFT-R.
                        mkspan = HEFT(dag, platform, priority_list=dag.top_sort)
                        info[ns][nw][ccr][h]["HEFT-R"].append(mkspan)
                        for rk in rks:
                            if rk == "F" or rk == "WF":
                                continue
                            avg_type = "WM" if rk == "W" else "HEFT" 
                            try:
                                mkspan = HEFT(dag, platform, cp_type=rk, avg_type=avg_type)  
                                info[ns][nw][ccr][h]["HEFT-" + rk].append(mkspan)
                            except KeyError:
                                info[ns][nw][ccr][h]["HEFT-" + rk].append(0.0)
                                pass 
                        # PEFT.
                        mkspan = PEFT(dag, platform)
                        info[ns][nw][ccr][h]["PEFT"].append(mkspan)
                        # Alternative processor selections.
                        for ps in pss: 
                            CCP = dag.conditional_critical_paths(cp_type=ps, lookahead=True)
                            priority_list = dag.conditional_critical_path_priorities(platform, CCP=CCP, cp_type=ps)
                            mkspan = HSM(dag, platform, cp_type=ps, CCP=CCP, priority_list=priority_list) 
                            info[ns][nw][ccr][h]["PEFT-" + ps].append(mkspan)
                            mks = HEFT(dag, platform, priority_list=priority_list)
                            info[ns][nw][ccr][h]["PEFT-" + ps + "-NPS"].append(mks)   
    elapsed = timer() - start
    print("\nTOTAL TIME: {} minutes".format(elapsed / 60), file=dest) 
                
# Save the info.
with open('{}/sz1000_info.dill'.format(results_path), 'wb') as handle:
    dill.dump(info, handle)                
                    
                
                
                
            
        