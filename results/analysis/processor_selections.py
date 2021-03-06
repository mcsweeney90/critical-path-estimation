#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis and plots for the processor selection section. 
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
summary_path = "../summaries/processor_selection"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/processor_selection"
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
all_attributes = ["MST", "HEFT", "PEFT", "PEFT-LB", "PEFT-M", "PEFT-WM", "PEFT-LB-NPS", "PEFT-M-NPS", "PEFT-WM-NPS"]
all_pss = ["LB", "M", "WM"]

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

def summarize(data, dest, pss):
    """
    Want direct comparison vs PEFT only.
    """
    
    # Calculate percentage degradations.
    pds = {ps : [] for ps in pss}
    pds["PEFT"] = []
    bests = {ps: 0 for ps in pss}
    bests["PEFT"] = 0
    for i, pm in enumerate(data["PEFT"]):
        if pm is None:
            continue
        best = pm
        for ps in pss:            
            best = min(best, data["PEFT-" + ps][i])
        # Now calculate the pds. 
        pd = (pm/best)*100 - 100
        pds["PEFT"].append(pd)
        if pm == best:
            bests["PEFT"] += 1
        for ps in pss:
            m = data["PEFT-" + ps][i]
            pd = (m/best)*100 - 100
            pds[ps].append(pd)  
            if m == best:
                bests[ps] += 1
    
    for ps in ["PEFT"] + pss:
        print("\nHEURISTIC: {}".format(ps), file=dest)
        print("APD: {}".format(np.mean(pds[ps])), file=dest)
        print("BEST INSTANCES (#, %): ({}, {})".format(bests[ps], (bests[ps]/len(pds[ps])*100)), file=dest)
        if ps == "PEFT":
            pfails = 0
            for m, mst in zip(data["PEFT"], data["MST"]):
                if m is None:
                    continue
                elif m > mst:
                    pfails += 1
            print("Failures: {}/{} ({} %)".format(pfails, len(data["PEFT"]), (pfails/len(data["PEFT"]))*100), file=dest)
            continue
        print("vs STANDARD PEFT", file=dest)
        reductions = []
        for m, pm in zip(data["PEFT-" + ps], data["PEFT"]):
            if pm is None:
                continue
            reductions.append(100 - (m/pm)*100)
        print("Reductions (%) (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
        better = sum(1 for r in reductions if r > 0)
        same = sum(1 for r in reductions if abs(r) < 1e-6)
        worse = sum(1 for r in reductions if r < 0)
        valid = len(reductions)
        print("(%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest) 
        fails, ct = 0, 0 
        for m, mst in zip(data["PEFT-" + ps], data["MST"]):
            if m is None:
                continue
            ct += 1
            if m > mst:
                fails += 1
        sign = "+" if fails > pfails else ""
        print("Change in failures : {}{}".format(sign, fails - pfails), file=dest)
        
        print("vs NO PROCESSOR SELECTION", file=dest)
        reductions = []
        for m, pm in zip(data["PEFT-" + ps], data["PEFT-" + ps + "-NPS"]):
            if pm is None or m is None:
                continue
            reductions.append(100 - (m/pm)*100)
        print("Reductions (%) (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
        better = sum(1 for r in reductions if r > 0)
        same = sum(1 for r in reductions if abs(r) < 1e-6)
        worse = sum(1 for r in reductions if r < 0)
        valid = len(reductions)
        print("(%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest) 
        nps_fails = 0 
        for m, mst in zip(data["PEFT-" + ps + "-NPS"], data["MST"]):
            if m is None:
                continue
            if m > mst:
                nps_fails += 1
        sign = "+" if fails > nps_fails else ""
        print("Change in failures : {}{}".format(sign, fails - nps_fails), file=dest)
        
# =============================================================================
# Human-readable(ish) summaries of the data.
# =============================================================================  

# Individual summaries for all 1 x 3 x 3 x 2 = 18 subsets (intended mostly as a reference since not easy to parse.)
with open("{}/all_subsets.txt".format(summary_path), "w") as dest:
    print("PEFT VARIANT PERFORMANCE FOR ALL SUBSETS (OF 5 x 180 = 900 DAGs) DEFINED BY PARAMETERS (n, q, CCR, h).", file=dest) 
    for n, q, b, h in all_param_combinations:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("n = {}, q = {}, CCR = {}, h = {}".format(n, q, b, h), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=[q], bs=[b], hs=[h], attrs=all_attributes)
        summarize(subset_info, dest, all_pss)
# By size.          
with open("{}/size_breakdown.txt".format(summary_path), "w") as dest:
    print("PEFT VARIANT PERFORMANCE BROKEN DOWN BY DAG SIZE n.", file=dest)     
    for n in sizes:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("n = {}".format(n), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=[n], qs=n_workers, bs=ccrs, hs=het_factors, attrs=all_attributes)
        summarize(subset_info, dest, all_pss)        
# By number of processors.          
with open("{}/processor_number_breakdown.txt".format(summary_path), "w") as dest:
    print("PEFT VARIANT PERFORMANCE BROKEN DOWN BY NUMBER OF PROCESSORS q.", file=dest)     
    for q in n_workers:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("q = {}".format(q), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors, attrs=all_attributes)
        summarize(subset_info, dest, all_pss)
# By CCR.          
with open("{}/ccr_breakdown.txt".format(summary_path), "w") as dest:
    print("PEFT VARIANT PERFORMANCE BROKEN DOWN BY CCR.", file=dest)     
    for b in ccrs:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("CCR = {}".format(b), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=[b], hs=het_factors, attrs=all_attributes)
        summarize(subset_info, dest, all_pss)
# By het factor.          
with open("{}/het_factor_breakdown.txt".format(summary_path), "w") as dest:
    print("PEFT VARIANT PERFORMANCE BROKEN DOWN BY HETEROGENEITY FACTOR h.", file=dest)     
    for h in het_factors:
        print("\n\n\n---------------------------------------------------------------------------------", file=dest)
        print("h = {}".format(h), file=dest)
        print("---------------------------------------------------------------------------------", file=dest)
        subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=[h], attrs=all_attributes)
        summarize(subset_info, dest, all_pss)
# Across the entire set.
with open("{}/complete.txt".format(summary_path), "w") as dest:
    print("PEFT VARIANT PERFORMANCE FOR ALL 43200 DAGs.", file=dest)   
    subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=het_factors, attrs=all_attributes)
    summarize(subset_info, dest, all_pss)
    
# =============================================================================
# Plots.
# =============================================================================

# Better instances than PEFT.
better_by_q = {ps:[] for ps in all_pss}
for q in n_workers:
    subset_info = get_subset_info(info, ns=sizes, qs=[q], bs=ccrs, hs=het_factors, attrs=all_attributes)    
    for ps in all_pss:
        reductions = []
        for m, pm in zip(subset_info["PEFT-" + ps], subset_info["PEFT"]):
            if pm is None:
                continue
            reductions.append(100 - (m/pm)*100)
        better = sum(1 for r in reductions if r > 0)
        valid = len(reductions)
        better_by_q[ps].append((better/valid)*100)

length, width = len(n_workers), 0.3
x = np.arange(length)
xlabels = n_workers
colors = {"LB": '#E24A33', "M" : '#348ABD', "WM" : '#988ED5'}

fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)

for j, ps in enumerate(all_pss):
    ax1.bar(x + j * width, better_by_q[ps], width, color=colors[ps], edgecolor='white', label=ps)             
ax1.set_xticks(x + width)
ax1.set_xticklabels(xlabels) 
ax1.set_xlabel("NUMBER OF PROCESSORS", labelpad=5)
ax1.set_ylabel("BETTER THAN PEFT (%)", labelpad=5)
ax1.legend(handlelength=3, handletextpad=0.4, ncol=3, loc='best', fancybox=True, facecolor='white') 
plt.savefig('{}/better_by_q'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 