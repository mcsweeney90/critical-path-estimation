#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and analysis for the benchmarking section.
"""

import dill, pathlib
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from collections import defaultdict

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

summary_path = "../summaries/benchmarking"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/benchmarking"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

with open('../sz100_info.dill', 'rb') as file:
    sz100_info = dill.load(file) 
with open('../sz1000_info.dill', 'rb') as file:
    sz1000_info = dill.load(file) 
info = {}
info[100] = sz100_info
info[1000] = sz1000_info

sizes = [100, 1000]
n_workers = [2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
all_param_combinations = list(it.product(*[sizes, n_workers, ccrs, het_factors]))
dag_names = list(info[100].keys())

# =============================================================================
# First an investigation...
# =============================================================================

# dag_names = list(d for d, _ in info[100].items()) 
# peft_mkspans = []
# ids = []
# for n in sizes:        
#         for q in n_workers:
#             for b in ccrs:
#                 for h in het_factors:
#                     for d in dag_names:
#                         if any(m is None for m in info[n][d][q][b][h]["PEFT"]):
#                             print("\nn = {}, q = {}, CCR = {}, h = {}, d = {}".format(n, q, b, h, d))
#                             print(info[n][d][q][b][h]["PEFT"])
#                             ids.append((n, q, b, h, "{}.dill".format(d)))
#                         peft_mkspans += info[n][d][q][b][h]["PEFT"]
# # Count how many...
# count = 0
# for p in peft_mkspans:
#     if p is None:
#         count += 1
# print("\nNumber of PEFT runs returning None: {}/{}".format(count, len(peft_mkspans)))

"""
Ran 1000 new runs with these DAGs and parameter combinations and couldn't reproduce the error. The PEFT function returns the variable mkspan, which is 
computed by the DAG makespan method. This uses Python's builtin max function which shouldn't ever return None, so cannot see where the problem is. Am going
to have to chalk this up to some unexplained bug either in the initial save or from dill. I'm satisfied that the PEFT function works in other cases and 
since only happened for very few (3) examples out of 43,200 think it best to just disregard these cases.  
"""  

# =============================================================================
# Human-readable(ish) summaries of the data.
# =============================================================================

def get_subset_info(info, ns, qs, bs, hs):
    """TODO: weird bug here, had to use defaultdict to avoid..."""
    set_info = defaultdict(list)
    param_combinations = list(it.product(*[ns, qs, bs, hs]))
    for n, q, b, h in param_combinations:
        for d in dag_names:
            for attr in ["CP", "MST", "HEFT", "PEFT"]:
                set_info[attr] += info[n][d][q][b][h][attr]
    #             try:
    #                 set_info[attr] += info[n][d][q][b][h][attr]
    #             except KeyError:
    #                 set_info[attr] = info[n][d][q][b][h][attr]
    return set_info    

def summarize_subset(subset_info, dest):
    for hr in ["HEFT", "PEFT"]:
        print("HEURISTIC: {}".format(hr), file=dest)
        slrs = []
        for m, cp in zip(subset_info[hr], subset_info["CP"]):
            if m is None:
                continue
            slrs.append(m/cp)
        print("SLRs (avg, best, worst) : ({}, {}, {})".format(np.mean(slrs), min(slrs), max(slrs)), file=dest)
        speedups = []
        for m, mst in zip(subset_info[hr], subset_info["MST"]):
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
        for m, hm in zip(subset_info[hr], subset_info["HEFT"]):
            if m is None:
                continue
            reductions.append(100 - (m/hm)*100)
        print("Reductions (%) vs HEFT (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
        better = sum(1 for r in reductions if r > 0)
        same = sum(1 for r in reductions if abs(r) < 1e-6)
        worse = sum(1 for r in reductions if r < 0)
        print("Comparison vs HEFT (%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest)    

# Individual summaries for all 2 x 4 x 3 x 2 = 48 subsets (intended mostly as a reference since not easy to parse.)
# with open("{}/complete.txt".format(summary_path), "w") as dest:
#     print("HEFT AND PEFT BENCHMARKING FOR ALL SUBSETS (OF 5 x 180 = 900 DAGs) DEFINED BY PARAMETERS (n, q, CCR, h).", file=dest) 
#     for n, q, b, h in all_param_combinations:
#         print("\nn = {}, q = {}, CCR = {}, h = {}".format(n, q, b, h), file=dest)
#         subset_info = get_subset_info(info, ns=[n], qs=[q], bs=[b], hs=[h])
#         summarize_subset(subset_info, dest)
# By size.          
with open("{}/size_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY DAG SIZE n.", file=dest)     
    for n in sizes:
        print("\nn = {}".format(n))
        print("\nn = {}".format(n), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=n_workers, bs=ccrs, hs=het_factors)
        print(len(subset_info["CP"]))
        summarize_subset(subset_info, dest)        
# By number of processors.          
with open("{}/processor_number_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY NUMBER OF PROCESSORS q.", file=dest)     
    for q in n_workers:
        print("\nq = {}".format(q))
        print("\nq = {}".format(q), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors)
        print(len(subset_info["CP"]))
        summarize_subset(subset_info, dest)
# By CCR.          
with open("{}/ccr_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY CCR.", file=dest)     
    for b in ccrs:
        print("\nCCR = {}".format(b))
        print("\nCCR = {}".format(b), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=[b], hs=het_factors)
        print(len(subset_info["CP"]))
        summarize_subset(subset_info, dest)
# By het factor.          
with open("{}/het_factor_breakdown.txt".format(summary_path), "w") as dest:
    print("HEFT AND PEFT BENCHMARKING BROKEN DOWN BY HETEROGENEITY FACTOR h.", file=dest)     
    for h in het_factors:
        print("\nh = {}".format(h))
        print("\nh = {}".format(h), file=dest)
        print("-----------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=[h])
        print(len(subset_info["CP"]))
        summarize_subset(subset_info, dest)
        
    
        
    
  
        
    
#     for n, q, b, h in param_combinations:
#         print("\nn = {}, q = {}, CCR = {}, h = {}".format(n, q, b, h), file=dest)
#         subset_info = {}
#         for d in dag_names:
#             for attr in ["CP", "MST", "HEFT", "PEFT"]:
#                 try:
#                     subset_info[attr] += info[n][d][q][b][h][attr]
#                 except KeyError:
#                     subset_info[attr] = info[n][d][q][b][h][attr]
#         # Summarise all info for the subset. 
#         for hr in ["HEFT", "PEFT"]:
#             print("HEURISTIC: {}".format(hr), file=dest)
#             slrs = []
#             for m, cp in zip(subset_info[hr], subset_info["CP"]):
#                 if m is None:
#                     continue
#                 slrs.append(m/cp)
#             print("SLRs (avg, best, worst) : ({}, {}, {})".format(np.mean(slrs), min(slrs), max(slrs)), file=dest)
#             speedups = []
#             for m, mst in zip(subset_info[hr], subset_info["MST"]):
#                 if m is None:
#                     continue
#                 speedups.append(mst/m)
#             print("Speedups (avg, best, worst) : ({}, {}, {})".format(np.mean(speedups), max(speedups), min(speedups)), file=dest)
#             valid = len(speedups)
#             failures = sum(1 for s in speedups if s < 1.0)
#             print("Failures (#, %): ({}, {})".format(failures, (failures/valid)*100), file=dest)
#             # Head-to-head.
#             if hr == "HEFT":
#                 continue
#             reductions = []
#             for m, hm in zip(subset_info[hr], subset_info["HEFT"]):
#                 if m is None:
#                     continue
#                 reductions.append(100 - (m/hm)*100)
#             print("Reductions (%) vs HEFT (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
#             better = sum(1 for r in reductions if r > 0)
#             same = sum(1 for r in reductions if abs(r) < 1e-6)
#             worse = sum(1 for r in reductions if r < 0)
#             print("Comparison vs HEFT (%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest)



# length, width = 3, 0.45
# x = np.arange(length)
# xlabels = [0.1, 1.0, 10]

# for sz in sizes:
#     fig = plt.figure(dpi=400) 
#     ax = fig.add_subplot(111, frameon=False)
#     # hide tick and tick label of the big axes
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     ax.set_xlabel("CCR", labelpad=5)
#     ax.set_ylabel("AVG. REDUCTION (%)", labelpad=10)
    
#     for i, nw in enumerate(n_workers):
        
#         ax1 = fig.add_subplot(len(n_workers) * 100 + 11 + i)
        
#         avg_reductions = {}
#         for m in methods:
#             avg_reductions[m] = [0.0, 0.0, 0.0]
#             for k, ccr in enumerate(ccrs):                
#                 for h in het_factors:
#                     avg_reductions[m][k] += np.mean(info[sz][nw][ccr][h][m]["reductions"])/len(het_factors)
#         j = 0
#         for m in methods:
#             bcolor = '#E24A33' if m == "UR" else '#348ABD'
#             ax1.bar(x + j * width, avg_reductions[m], width, color=bcolor, edgecolor='white', label=m)
#             j += 1               
#         ax1.set_xticks(x + (1/2) * width)
#         if i < 3:
#             ax1.set_xticklabels([]) 
#             if i == 0:
#                 ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white')                
#         else:
#             ax1.set_xticklabels(xlabels)        
#         ax1.set_title("{} PROCESSORS".format(nw), color="black", family='serif')
        
#     plt.savefig('plots/{}_{}tasks_reductions'.format(name, sz), bbox_inches='tight') 
#     plt.close(fig) 

