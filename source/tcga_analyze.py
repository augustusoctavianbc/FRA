import numpy as np
import pandas as pd
from scipy.stats import rankdata
import rankutils
from rankutils import lehmer_max_consensus
from rankutils import lehmer_max_consensus_l
from rankutils import average_consensus
from rankutils import bin_average_consensus
from rankutils import footrule_consensus
from rankutils import calc_dist_from_aggregate
from rankutils import perm_2_lehmer
from rankutils import calc_threshold

import matplotlib.pyplot as plt
import time

t1 = time.time()
np.random.seed(42)


def aggregate_communicate_tcga(cancer):
    
    df = pd.read_pickle('../data/tcga/'+cancer+'_combine_dataframe')
    samples = len(df)

    rank_lists = []
    for itr in range(samples):
        rank_lists.append(len(df.expression.iloc[itr]) +1 - rankdata(df.expression.iloc[itr].to_list(), method='ordinal'))
        
    #these are in two line notation because of the use of rankdata function from scipy.stats
    rank_lists = np.array(rank_lists)
    
    #rows shuffled for randomized sample allocation
    np.random.shuffle(rank_lists)
    #rank_lists = rank_lists[0:500]
    
    rank_lists_l = []
    for p in rank_lists:
        rank_lists_l.append(perm_2_lehmer(p))
    rank_lists_l = np.array(rank_lists_l)
    
    num_clients = 10
    samples_per_list = [itr for itr in range(1,int(len(rank_lists)/num_clients)+1,1)]
    
    #samples_per_central = [num_clients*itr for itr in range(1,int(len(sushi_perms)/num_clients)+1,1)]
    
    phi=0.6
    m=len(rank_lists[0])
    threshold = calc_threshold(phi,m,100000)
    
    lehmer_v_samples = []
    direct_v_samples = []
    footrule_v_samples = []
    
    lehmer_v_central = []
    direct_v_central = []
    footrule_v_central = []
    
    for samples_per in samples_per_list:
        client_lehmer_agg = []
        client_direct_agg = []
        client_footrule_agg = []
        
        for client in range(num_clients):
            
            client_perms = rank_lists[client*samples_per:(client+1)*samples_per, :]
            client_perms_l = rank_lists_l[client*samples_per:(client+1)*samples_per, :]
            
            client_lehmer_agg.append(lehmer_max_consensus_l(client_perms_l))
            client_direct_agg.append(bin_average_consensus(client_perms, threshold))
            client_footrule_agg.append(footrule_consensus(client_perms))
            
        lehmer_agg_agg = lehmer_max_consensus(client_lehmer_agg)
        direct_agg_agg = average_consensus(client_direct_agg)
        footrule_agg_agg = footrule_consensus(client_footrule_agg)
        
        lehmer_agg_agg_dist = calc_dist_from_aggregate(rank_lists, lehmer_agg_agg)
        direct_agg_agg_dist = calc_dist_from_aggregate(rank_lists, direct_agg_agg)
        footrule_agg_agg_dist = calc_dist_from_aggregate(rank_lists, footrule_agg_agg)
        
        
        print(samples_per, lehmer_agg_agg_dist, direct_agg_agg_dist, footrule_agg_agg_dist, time.time()-t1)
        lehmer_v_samples.append(lehmer_agg_agg_dist)
        direct_v_samples.append(direct_agg_agg_dist)
        footrule_v_samples.append(footrule_agg_agg_dist)
        
        
        #centralized aggregates
        lehmer_central = lehmer_max_consensus_l(rank_lists_l[0:num_clients*samples_per,:])
        lehmer_central_dist = calc_dist_from_aggregate(rank_lists, lehmer_central)

        direct_central = average_consensus(rank_lists[0:num_clients*samples_per,:])
        direct_central_dist = calc_dist_from_aggregate(rank_lists, direct_central)

        footrule_central = footrule_consensus(rank_lists[0:num_clients*samples_per,:])
        footrule_central_dist = calc_dist_from_aggregate(rank_lists, footrule_central)

        
        lehmer_v_central.append(lehmer_central_dist)
        direct_v_central.append(direct_central_dist)
        footrule_v_central.append(footrule_central_dist)
    
        
    fig, ax1 = plt.subplots(figsize=(8,6))
    #ax1 = fig.add_subplot(111)

    ax1.plot(samples_per_list, lehmer_v_samples, label='Lehmer coordinate majority', color='tab:blue')
    ax1.plot(samples_per_list, direct_v_samples, label='Borda coordinate quantization', color='tab:orange')
    ax1.plot(samples_per_list, footrule_v_samples, label='Footrule bipartite matching', color='tab:green')

    ax1.plot(samples_per_list, lehmer_v_central, color='tab:blue', linestyle='--')
    ax1.plot(samples_per_list, direct_v_central, color='tab:orange', linestyle='--')
    ax1.plot(samples_per_list, footrule_v_central, color='tab:green', linestyle='--')

    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([int(x * num_clients) for x in ax1.get_xticks()])
    
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax1.set_xlabel('Samples at each client', fontsize=14)
    ax1.set_ylabel('Centralized objective', fontsize=14)
    ax2.set_xlabel('Total number of samples (with possible repetitions)', fontsize=14)
    ax1.legend(fontsize=14)
    ax1.set_title(cancer, fontsize=14)

    plt.savefig('../figures/'+cancer+'_samples06_10.png', bbox_inches='tight')
    plt.close()
    #plt.show()
    return 0

if __name__ == "__main__":
    aggregate_communicate_tcga(cancer = 'LUSC')
    aggregate_communicate_tcga(cancer = 'LUAD')