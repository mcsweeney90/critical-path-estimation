#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis and plots for the benchmarking section.
"""

import dill, pathlib
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from collections import defaultdict

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.

####################################################################################################

# Destinations to save summaries and generated plots.
summary_path = "../summaries/benchmarking"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/benchmarking"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

# Combine the info dicts for different sizes.
with open('../sz100_info.dill', 'rb') as file:
    sz100_info = dill.load(file) 
with open('../sz1000_info.dill', 'rb') as file:
    sz1000_info = dill.load(file) 
info = {}
info[100] = sz100_info
info[1000] = sz1000_info

# Parameters.
sizes = [100, 1000]
n_workers = [2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]

# These are useful later. 
all_param_combinations = list(it.product(*[sizes, n_workers, ccrs, het_factors]))
dag_names = list(info[100].keys())

# =============================================================================
"""
First, an investigation. I noticed that PEFT apparently returned None for 3/43200 DAGs, all with n = 100. 
The PEFT function returns the variable mkspan, which is computed by the DAG makespan method. This uses Python's builtin 
max function which shouldn't ever return None, so not sure where the problem is. For each of the 3 DAGs, I did 1000 new runs
with the same parameter combination and couldn't reproduce the error so am going to have to chalk this up to some unexplained 
bug either in the initial save or from dill. I'm satisfied that the PEFT function works in other cases and 
since only happened for very few examples (3/43,200) think it best to just disregard these cases. 
"""
# for n in sizes:        
#         for q in n_workers:
#             for b in ccrs:
#                 for h in het_factors:
#                     for d in dag_names:
#                         if any(m is None for m in info[n][d][q][b][h]["PEFT"]):
#                             print("\nn = {}, q = {}, CCR = {}, h = {}, d = {}".format(n, q, b, h, d))
#                             print(info[n][d][q][b][h]["PEFT"])

# =============================================================================


def get_subset_info(info, ns, qs, bs, hs):
    """
    Get all of the info for subset of the DAGs defined by n in ns, q in qs, etc.     
    """
    set_info = defaultdict(list)
    param_combinations = list(it.product(*[ns, qs, bs, hs]))
    for n, q, b, h in param_combinations:
        for d in dag_names:
            for attr in ["CP", "MST", "HEFT", "PEFT"]:
                set_info[attr] += info[n][d][q][b][h][attr]
    return set_info    

def summarize(data, dest):
    """Summarize things that we want for the data."""
    for hr in ["HEFT", "PEFT"]:
        print("HEURISTIC: {}".format(hr), file=dest)
        slrs = []
        for m, cp in zip(data[hr], data["CP"]):
            if m is None:
                continue
            slrs.append(m/cp)
        print("SLRs (avg, best, worst) : ({}, {}, {})".format(np.mean(slrs), min(slrs), max(slrs)), file=dest)
        speedups = []
        for m, mst in zip(data[hr], data["MST"]):
            if m is None:
                continue
            speedups.append(mst/m)
        print("Speedups (avg, best, worst) : ({}, {}, {})".format(np.mean(speedups), max(speedups), min(speedups)), file=dest)
        valid = len(speedups)
        failures = sum(1 for s in speedups if s < 1.0)
        print("Failures (#, %): ({}, {})".format(failures, (failures/valid)*100), file=dest)
        # Head-to-head.
        if hr == "HEFT":
            continue
        reductions = []
        for m, hm in zip(data[hr], data["HEFT"]):
            if m is None:
                continue
            reductions.append(100 - (m/hm)*100)
        print("Reductions (%) vs HEFT (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
        better = sum(1 for r in reductions if r > 0)
        same = sum(1 for r in reductions if abs(r) < 1e-6)
        worse = sum(1 for r in reductions if r < 0)
        print("Comparison vs HEFT (%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest)    

# =============================================================================
# Human-readable(ish) summaries of the data.
# =============================================================================       

# Individual summaries for all 2 x 4 x 3 x 2 = 48 subsets (intended mostly as a reference since not easy to parse.)
with open("{}/all_subsets.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING FOR ALL SUBSETS (OF 5 x 180 = 900 DAGs) DEFINED BY PARAMETERS (n, q, CCR, h).", file=dest) 
    for n, q, b, h in all_param_combinations:
        print("\nn = {}, q = {}, CCR = {}, h = {}".format(n, q, b, h), file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=[q], bs=[b], hs=[h])
        summarize(subset_info, dest)
# By size.          
with open("{}/size_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY DAG SIZE n.", file=dest)     
    for n in sizes:
        print("\nn = {}".format(n), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=n_workers, bs=ccrs, hs=het_factors)
        summarize(subset_info, dest)        
# By number of processors.          
with open("{}/processor_number_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY NUMBER OF PROCESSORS q.", file=dest)     
    for q in n_workers:
        print("\nq = {}".format(q), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors)
        summarize(subset_info, dest)
# By CCR.          
with open("{}/ccr_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY CCR.", file=dest)     
    for b in ccrs:
        print("\nCCR = {}".format(b), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=[b], hs=het_factors)
        summarize(subset_info, dest)
# By het factor.          
with open("{}/het_factor_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY HETEROGENEITY FACTOR h.", file=dest)     
    for h in het_factors:
        print("\nh = {}".format(h), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=[h])
        summarize(subset_info, dest)
# Across the entire set.
with open("{}/complete.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING SUMMARY FOR THE ENTIRE DAG SET.", file=dest)   
    subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=het_factors)
    summarize(subset_info, dest)

# =============================================================================
# Plots.
# =============================================================================

# SLR.
slr_by_worker = {"HEFT":[], "PEFT":[]}
for q in n_workers:
    subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors)
    for hr in ["HEFT", "PEFT"]:
        slrs = []
        for m, cp in zip(subset_info[hr], subset_info["CP"]):
            if m is None:
                continue
            slrs.append(m/cp)
        slr_by_worker[hr].append(np.mean(slrs))

length, width = len(n_workers), 0.45
x = np.arange(length)
xlabels = n_workers

fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)
ax1.bar(x, slr_by_worker["HEFT"], width, color='#E24A33', edgecolor='white', label="HEFT")
ax1.bar(x + width, slr_by_worker["PEFT"], width, color='#348ABD', edgecolor='white', label="PEFT")             
ax1.set_xticks(x + (1/2) * width)
ax1.set_xticklabels(xlabels) 
ax1.set_xlabel("NUMBER OF PROCESSORS", labelpad=5)
ax1.set_ylabel("SCHEDULE LENGTH RATIO", labelpad=5)
ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white') 
plt.savefig('{}/slr_by_q'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 

# Speedup.
speedup_by_worker = {"HEFT":[], "PEFT":[]}
for q in n_workers:
    subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors)
    for hr in ["HEFT", "PEFT"]:
        speedups = []
        for m, mst in zip(subset_info[hr], subset_info["MST"]):
            if m is None:
                continue
            speedups.append(mst/m)
        speedup_by_worker[hr].append(np.mean(speedups))

length, width = len(n_workers), 0.45
x = np.arange(length)
xlabels = n_workers

fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)
ax1.bar(x, speedup_by_worker["HEFT"], width, color='#E24A33', edgecolor='white', label="HEFT")
ax1.bar(x + width, speedup_by_worker["PEFT"], width, color='#348ABD', edgecolor='white', label="PEFT")             
ax1.set_xticks(x + (1/2) * width)
ax1.set_xticklabels(xlabels) 
ax1.set_xlabel("NUMBER OF PROCESSORS", labelpad=5)
ax1.set_ylabel("SPEEDUP", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white') 
plt.savefig('{}/speedup_by_q'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 

