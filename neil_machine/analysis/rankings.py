#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and analysis for the alternative HEFT rankings section. 
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
plt.rcParams['font.weight'] = 'bold' # Don't know if I always want this...
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

summary_path = "../summaries/rankings"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/rankings"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

with open('../sz100_info.dill', 'rb') as file:
    sz100_info = dill.load(file) 
with open('../sz1000_info.dill', 'rb') as file:
    sz1000_info = dill.load(file) 
info = {}
info[100] = sz100_info
info[1000] = sz1000_info

sizes = [100]#[100, 1000]
n_workers = [2, 4, 8] #[2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [1.0, 2.0]
all_param_combinations = list(it.product(*[sizes, n_workers, ccrs, het_factors]))
dag_names = list(info[100].keys())
all_attributes = ["MST", "HEFT", "HEFT-R", "HEFT-LB", "HEFT-W", "HEFT-F", "HEFT-WF"]
all_rks = ["R", "LB", "W", "F", "WF"]

####################################################################################################

def get_subset_info(info, ns, qs, bs, hs, attrs):
    """
    TODO: weird bug here, had to use defaultdict to avoid.
    Problem was that I was using a try statement to create keys and I had another variable outside with a similar name...    
    """
    set_info = defaultdict(list)
    param_combinations = list(it.product(*[ns, qs, bs, hs]))
    for n, q, b, h in param_combinations:
        for d in dag_names:
            for attr in attrs: # TODO. Insert Nones for empty lists?
                set_info[attr] += info[n][d][q][b][h][attr]
                # if not info[n][d][q][b][h][attr]:
                #     set_info[attr] += [None] * 5
                # else:
                #     set_info[attr] += info[n][d][q][b][h][attr]
    return set_info  

def summarize(data, dest, rks):
    """Want direct comparison vs HEFT only."""
    
    pds = {rk : [] for rk in rks}
    pds["U"] = []
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
        for rk in rks:
            m = data["HEFT-" + rk][i]
            pd = (m/best)*100 - 100
            pds[rk].append(pd)    
    
    for rk in rks + ["U"]:
        print("RANKING: {}".format(rk), file=dest)
        print("APD: {}".format(np.mean(pds[rk])), file=dest)
        if rk == "U":
            continue
        reductions = []
        for m, hm in zip(data["HEFT-" + rk], data["HEFT"]):
            if m is None:
                continue
            reductions.append(100 - (m/hm)*100)
        print("Reductions (%) vs HEFT (avg, best, worst) : ({}, {}, {})".format(np.mean(reductions), max(reductions), min(reductions)), file=dest)
        better = sum(1 for r in reductions if r > 0)
        same = sum(1 for r in reductions if abs(r) < 1e-6)
        worse = sum(1 for r in reductions if r < 0)
        valid = len(reductions)
        print("Comparison vs HEFT (%better, %same, %worse): ({}, {}, {})".format((better/valid)*100, (same/valid)*100, (worse/valid)*100), file=dest) 

# =============================================================================
# Human-readable(ish) summaries of the data.
# =============================================================================  

# Individual summaries for all 2 x 4 x 3 x 2 = 48 subsets (intended mostly as a reference since not easy to parse.)
with open("{}/all_subsets.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORAMNCE VS DEFAULT FOR ALL SUBSETS (OF 5 x 180 = 900 DAGs) DEFINED BY PARAMETERS (n, q, CCR, h).", file=dest) 
    for n, q, b, h in all_param_combinations:
        print("\nn = {}, q = {}, CCR = {}, h = {}".format(n, q, b, h), file=dest)
        if (n == 1000 or q == 16):
            attributes = all_attributes[:-2]
            rks = all_rks[:-2]
        else:
            attributes = all_attributes
            rks = all_rks
        subset_info = get_subset_info(info, ns=[n], qs=[q], bs=[b], hs=[h], attrs=attributes)
        summarize(subset_info, dest, rks)
with open("{}/f_and_wf_complete.txt".format(summary_path), "w") as dest:
    print("ALTERNATIVE HEFT RANKING PERFORAMNCE FOR ALL 16200 DAGs F AND WF WERE RUN ON.", file=dest) 
    subset_info = get_subset_info(info, ns=sizes, qs=n_workers, bs=ccrs, hs=het_factors, attrs=all_attributes)
    summarize(subset_info, dest, all_rks)