#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run comparison of modified HEFT and PEFT heuristics for size 1000 DAGs from the STG.
"""

import dill, os
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Platform, HEFT, PEFT, HSM

# Location to DAGs - may need to be changed depending on where script is located.
n = 1000
dag_path = '../graphs/STG/{}'.format(n)

# Parameters.
n_workers = [2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
rks = ["LB", "W", "F", "WF"]
pss = ["LB", "M", "WM"]

# Initialize info dict - bit ugly to do it this way but prevents KeyErrors. 
info = {}
for name in os.listdir('{}'.format(dag_path)):
    ns = name[:8]
    info[ns] = {}
    for q in n_workers:
        info[ns][q] = {}
        for b in ccrs:
            info[ns][q][b] = {}
            for h in het_factors:
                info[ns][q][b][h] = {}
                info[ns][q][b][h]["MST"] = []
                info[ns][q][b][h]["CP"] = []
                info[ns][q][b][h]["HEFT"] = []
                info[ns][q][b][h]["HEFT-R"] = []
                for rk in rks:
                    info[ns][q][b][h]["HEFT-" + rk] = []
                info[ns][q][b][h]["PEFT"] = []
                for ps in pss:
                    info[ns][q][b][h]["PEFT-" + ps] = []
                    info[ns][q][b][h]["PEFT-" + ps + "-NPS"] = []
                    
# Run the actual comparison.         
with open("sz1000_timing.txt", "w") as dest:
    start = timer()     # Time the entire comparison.
    # Iterate over DAG directory.
    for name in os.listdir('{}'.format(dag_path)):
        ns = name[:8]
        # Load DAG topology.
        with open('{}/{}'.format(dag_path, name), 'rb') as file:
            dag = dill.load(file)
        for q in n_workers:
            # Create platform.
            platform = Platform(q, name="{}P".format(q))
            for b in ccrs:
                for h in het_factors:
                    for _ in range(5):
                        # Set DAG costs.
                        dag.set_costs(platform, target_ccr=b, method="R", het_factor=h)
                        # Minimal serial time.
                        mst = dag.minimal_serial_time()
                        info[ns][q][b][h]["MST"].append(mst)
                        # Optimal critical path/lower bound.
                        OCP = dag.conditional_critical_paths(direction="downward", cp_type="LB")
                        cp = max(min(OCP[task.ID][p] for p in OCP[task.ID]) for task in dag.graph if task.exit)
                        info[ns][q][b][h]["CP"].append(cp)
                        # HEFT.
                        mkspan = HEFT(dag, platform)
                        info[ns][q][b][h]["HEFT"].append(mkspan)
                        # HEFT-R.
                        mkspan = HEFT(dag, platform, priority_list=dag.top_sort)
                        info[ns][q][b][h]["HEFT-R"].append(mkspan)
                        for rk in rks:
                            if rk == "F" or rk == "WF":
                                continue        # Results for n = 100 suggest F and WF not worthwhile.
                            avg_type = "WM" if rk == "W" else "HEFT" 
                            try:
                                mkspan = HEFT(dag, platform, cp_type=rk, avg_type=avg_type)  
                                info[ns][q][b][h]["HEFT-" + rk].append(mkspan)
                            except KeyError:    # Catch rare error that sometimes occurs for WF (think due to rounding error).
                                info[ns][q][b][h]["HEFT-" + rk].append(0.0)
                                pass 
                        # PEFT.
                        mkspan = PEFT(dag, platform)
                        info[ns][q][b][h]["PEFT"].append(mkspan)
                        # Alternative processor selections.
                        for ps in pss: 
                            CCP = dag.conditional_critical_paths(cp_type=ps, lookahead=True)
                            priority_list = dag.conditional_critical_path_priorities(platform, CCP=CCP, cp_type=ps)
                            mkspan = HSM(dag, platform, cp_type=ps, CCP=CCP, priority_list=priority_list) 
                            info[ns][q][b][h]["PEFT-" + ps].append(mkspan)
                            # Without the PEFT-like processor selection phase.
                            mks = HEFT(dag, platform, priority_list=priority_list)
                            info[ns][q][b][h]["PEFT-" + ps + "-NPS"].append(mks)   
    elapsed = timer() - start
    print("\nTOTAL TIME: {} minutes".format(elapsed / 60), file=dest) 
                
# Save the info dict.
with open('sz1000_info.dill', 'wb') as handle:
    dill.dump(info, handle)                
                    
                
                
                
            
        