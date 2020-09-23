#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis and plots for the (small-scale) alternative HEFT rankings section. 
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
summary_path = "../summaries/small_scale_rankings"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/small_scale_rankings"
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
sizes = [100]
n_workers = [2, 4, 8] 
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]

# These are useful later. 
all_param_combinations = list(it.product(*[sizes, n_workers, ccrs, het_factors]))
dag_names = list(info[100].keys())
all_attributes = ["MST", "HEFT", "HEFT-R", "HEFT-LB", "HEFT-W", "HEFT-F", "HEFT-WF"]
all_rks = ["R", "LB", "W", "F", "WF"]

####################################################################################################

def get_subset_info(info, ns, qs, bs, hs, attrs):
    """
    Get all of the info for subset of the DAGs defined by n in ns, q in qs, etc.    
    """
    set_info = defaultdict(list)
    param_combinations = list(it.product(*[ns, qs, bs, hs]))
    for n, q, b, h in param_combinations:
        for d in dag_names:
            for attr in attrs: 
                set_info[attr] += info[n][d][q][b][h][attr]
    return set_info  

def summarize(data, dest, rks):
    """Want direct comparison vs HEFT only."""
    
    # Calculate percentage degradations.
    pds = {rk : [] for rk in rks}
    pds["U"] = []
    bests = {rk: 0 for rk in rks}
    bests["U"] = 0
    for i, hm in enumerate(data["HEFT"]):
        best, valid = hm, True
        for rk in rks:
            if data["HEFT-" + rk][i] is None or data["HEFT-" + rk][i] == 0.0:
                valid = False
                break
            best = min(best, data["HEFT-" + rk][i])
        if not valid:
            continue
        # Now calculate the pds. 
        pd = (hm/best)*100 - 100
        pds["U"].append(pd)
        if hm == best:
            bests["U"] += 1
        for rk in rks:
            m = data["HEFT-" + rk][i]
            pd = (m/best)*100 - 100
            pds[rk].append(pd)  
            if m == best:
                bests[rk] += 1
    
    for rk in ["U"] + rks:
        print("\nRANKING: {}".format(rk), file=dest)
        print("APD: {}".format(np.mean(pds[rk])), file=dest)
        print("BEST INSTANCES (#, %): ({}, {})".format(bests[rk], (bests[rk]/len(pds[rk])*100)), file=dest)
        if rk == "U":
            ufails = 0
            for m, mst in zip(data["HEFT"], data["MST"]):
                if m > mst:
                    ufails += 1
            print("Failures: {}/{} ({} %)".format(ufails, len(data["HEFT"]), (ufails/len(data["HEFT"]))*100), file=dest)
            continue
        print("vs STANDARD U RANKING", file=dest)
        reductions = []
        for m, hm in zip(data["HEFT-" + rk], data["HEFT"]):
            if m is None:
                continue
            reductions.append(100 - (m/hm)*100)
        print("Reductions (%) (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
        better = sum(1 for r in reductions if r > 0)
        same = sum(1 for r in reductions if abs(r) < 1e-6)
        worse = sum(1 for r in reductions if r < 0)
        valid = len(reductions)
        print("(%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest) 
        fails, ct = 0, 0 
        for m, mst in zip(data["HEFT-" + rk], data["MST"]):
            if m is None:
                continue
            ct += 1
            if m > mst:
                fails += 1
        sign = "+" if fails > ufails else ""
        print("Change in failures : {}{}".format(sign, fails - ufails), file=dest)
        
# =============================================================================
# Human-readable(ish) summaries of the data.
# =============================================================================  

# Individual summaries for all 1 x 3 x 3 x 2 = 18 subsets (intended mostly as a reference since not easy to parse.)
with open("{}/all_subsets.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORMANCE FOR ALL SUBSETS (OF 5 x 180 = 900 DAGs) DEFINED BY PARAMETERS (n, q, CCR, h).", file=dest) 
    for n, q, b, h in all_param_combinations:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("n = {}, q = {}, CCR = {}, h = {}".format(n, q, b, h), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=[q], bs=[b], hs=[h], attrs=all_attributes)
        summarize(subset_info, dest, all_rks)
# By size.          
with open("{}/size_breakdown.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORMANCE BROKEN DOWN BY DAG SIZE n.", file=dest)     
    for n in sizes:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("n = {}".format(n), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=n_workers, bs=ccrs, hs=het_factors, attrs=all_attributes)
        summarize(subset_info, dest, all_rks)        
# By number of processors.          
with open("{}/processor_number_breakdown.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORMANCE BROKEN DOWN BY NUMBER OF PROCESSORS q.", file=dest)     
    for q in n_workers:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("q = {}".format(q), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors, attrs=all_attributes)
        summarize(subset_info, dest, all_rks)
# By CCR.          
with open("{}/ccr_breakdown.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORMANCE BROKEN DOWN BY CCR.", file=dest)     
    for b in ccrs:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("CCR = {}".format(b), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=[b], hs=het_factors, attrs=all_attributes)
        summarize(subset_info, dest, all_rks)
# By het factor.          
with open("{}/het_factor_breakdown.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORMANCE BROKEN DOWN BY HETEROGENEITY FACTOR h.", file=dest)     
    for h in het_factors:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("h = {}".format(h), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=[h], attrs=all_attributes)
        summarize(subset_info, dest, all_rks)
# Across the entire set.
with open("{}/complete.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORMANCE FOR ALL 16200 DAGs F AND WF WERE RUN ON.", file=dest)   
    subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=het_factors, attrs=all_attributes)
    summarize(subset_info, dest, all_rks)
    
# =============================================================================
# Plots.
# =============================================================================

# APD.
apd_by_ccr = {rk:[] for rk in all_rks}
apd_by_ccr["U"] = []
for b in ccrs:
    subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=[b], hs=het_factors, attrs=all_attributes)
    pds = {rk : [] for rk in all_rks}
    pds["U"] = []
    for i, hm in enumerate(subset_info["HEFT"]):
        best, valid = hm, True
        for rk in all_rks:
            if subset_info["HEFT-" + rk][i] is None or subset_info["HEFT-" + rk][i] == 0.0:
                valid = False
                break
            best = min(best, subset_info["HEFT-" + rk][i])
        if not valid:
            continue
        # Now calculate the pds. 
        pd = (hm/best)*100 - 100
        pds["U"].append(pd)
        for rk in all_rks:
            m = subset_info["HEFT-" + rk][i]
            pd = (m/best)*100 - 100
            pds[rk].append(pd) 
    for rk in ["U"] + all_rks:
        apd_by_ccr[rk].append(np.mean(pds[rk]))

length, width = len(ccrs), 0.15
x = np.arange(length)
xlabels = ccrs
colors = {"U": '#E24A33', "R" : '#348ABD', "LB" : '#988ED5', "W" : '#FBC15E', "F" : '#8EBA42', "WF" : '#FFB5B8'}

fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)

for j, rk in enumerate(["U"] + all_rks):
    ax1.bar(x + j * width, apd_by_ccr[rk], width, color=colors[rk], edgecolor='white', label=rk)             
ax1.set_xticks(x + (5/2) * width)
ax1.set_xticklabels(xlabels) 
ax1.set_xlabel("COMPUTATION TO COMMUNICATION RATIO (CCR)", labelpad=5)
ax1.set_ylabel("AVG. PERCENTAGE DEGRADATION (APD)", labelpad=5)
ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
plt.savefig('{}/apd_by_ccr'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 