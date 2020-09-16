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

# =============================================================================
# Print a human-readable summary of all the information.
# =============================================================================

with open("{}/complete.txt".format(summary_path), "w") as dest:
    print("HUMAN-READABLE HEFT AND PEFT BENCHMARKING SUMMARY.", file=dest)
    
    dag_names = list(d for d, _ in info[100].items())
    
    for q in n_workers:
        print("\n----------------", file=dest)
        print("{} PROCESSORS".format(q), file=dest)
        print("----------------", file=dest)
        
        # First for each combination...        
        for n in sizes:
            for b in ccrs:
                for h in het_factors:
                    print("\nn = {}, CCR = {}, h = {}".format(n, b, h), file=dest)
                    cps, msts, heft_mkspans, peft_mkspans = [], [], [], []
                    for d in dag_names:
                        cps += info[n][d][q][b][h]["CP"]
                        msts += info[n][d][q][b][h]["MST"]
                        heft_mkspans += info[n][d][q][b][h]["HEFT"]
                        peft_mkspans += info[n][d][q][b][h]["PEFT"]
                    # Now compute the SLRs.
                    heft_slrs = list(m/c for m, c in zip(heft_mkspans, cps))
                    print("HEFT SLRs (avg, best, worst) : ({}, {}, {})".format(np.mean(heft_slrs), min(heft_slrs), max(heft_slrs)), file=dest)
        



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

