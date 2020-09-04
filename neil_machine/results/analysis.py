#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and analysis.
"""

import dill
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

name = "heft_benchmarking_stg"

with open('{}.dill'.format(name), 'rb') as file:
    info = dill.load(file) 
    
sizes = [100]
n_workers = [2, 4, 8, 16]
ccrs = [0.1, 1, 10]
het_factors = [0.5, 1.0, 1.5, 2.0]
methods = ["R", "UR"]

length, width = 3, 0.45
x = np.arange(length)
xlabels = [0.1, 1.0, 10]

for sz in sizes:
    fig = plt.figure(dpi=400) 
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("CCR", labelpad=5)
    ax.set_ylabel("AVG. REDUCTION (%)", labelpad=10)
    
    for i, nw in enumerate(n_workers):
        
        ax1 = fig.add_subplot(len(n_workers) * 100 + 11 + i)
        
        avg_reductions = {}
        for m in methods:
            avg_reductions[m] = [0.0, 0.0, 0.0]
            for k, ccr in enumerate(ccrs):                
                for h in het_factors:
                    avg_reductions[m][k] += np.mean(info[sz][nw][ccr][h][m]["reductions"])/len(het_factors)
        j = 0
        for m in methods:
            bcolor = '#E24A33' if m == "UR" else '#348ABD'
            ax1.bar(x + j * width, avg_reductions[m], width, color=bcolor, edgecolor='white', label=m)
            j += 1               
        ax1.set_xticks(x + (1/2) * width)
        if i < 3:
            ax1.set_xticklabels([]) 
            if i == 0:
                ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white')                
        else:
            ax1.set_xticklabels(xlabels)        
        ax1.set_title("{} PROCESSORS".format(nw), color="black", family='serif')
        
    plt.savefig('plots/{}_{}tasks_reductions'.format(name, sz), bbox_inches='tight') 
    plt.close(fig) 

