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
from rankutils import invert_perm
from rankutils import calc_threshold

import matplotlib.pyplot as plt
import time

t1 = time.time()
np.random.seed(42)



def aggregate_communicate_sushi4():
    #number of samples increases in centralized version as well
    #distance at each iteration is calculated wrt all the samples
    #also called the Kemeney distance
    
    data_file = '../data/sushi3-2016/sushi3a.5000.10.order'
    user_file = '../data/sushi3-2016/sushi3.udata'
    user_data = pd.read_csv(user_file, sep='\t', header=None)
    sushi_perms = np.loadtxt(data_file, skiprows=1, dtype='int')
    sushi_perms = sushi_perms[:,2:]
    sushi_perms+=1
    

    #convert preference lists (1-line notation )to ranked lists (in 2-line notation)
    sushi_perms_2l = []
    for p in sushi_perms:
        sushi_perms_2l.append(invert_perm(p))
    
    sushi_perms_2l = np.array(sushi_perms_2l)
    
    num_clients = 10
    #samples_per_list = [itr for itr in range(1,int(len(sushi_perms_2l)/num_clients)+1,1)]
    #samples_frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    samples_frac = [x/200 for x in range(3,201,1)]
    #samples_per_central = [num_clients*itr for itr in range(1,int(len(sushi_perms)/num_clients)+1,1)]
    
    all_region_idx = {}
    for client in range(num_clients):
        all_region_idx[client] = user_data[user_data[8]==client].index.to_list()
        
    #we ignore the 19 samples for foreign labels
    all_jpn_idx = user_data[user_data[8]!=10].index.to_list()
    
    phi=0.6
    m=len(sushi_perms_2l[0])
    threshold = calc_threshold(phi,m,100000)
    
    lehmer_v_samples = []
    direct_v_samples = []
    footrule_v_samples = []
    
    lehmer_v_central = []
    direct_v_central = []
    footrule_v_central = []
    
    for samples_per in samples_frac:
        client_lehmer_agg = []
        client_direct_agg = []
        client_footrule_agg = []
        central_perms = []
        
        for client in range(num_clients):
            
            #take a fraction of rankings at each client
            client_perms_idx = all_region_idx[client][:int((samples_per)*len(all_region_idx[client]))]
            client_perms = sushi_perms_2l[client_perms_idx, :]
            
            #keep a record of all rankings to evalaute centralized performance
            central_perms.extend(client_perms)
            
            client_lehmer_agg.append(lehmer_max_consensus(client_perms))
            client_direct_agg.append(bin_average_consensus(client_perms, threshold))
            client_footrule_agg.append(footrule_consensus(client_perms))
            
            
        lehmer_agg_agg = lehmer_max_consensus(client_lehmer_agg)
        direct_agg_agg = average_consensus(client_direct_agg)
        footrule_agg_agg = footrule_consensus(client_footrule_agg)
        
        
        lehmer_agg_agg_dist = calc_dist_from_aggregate(sushi_perms_2l[all_jpn_idx], lehmer_agg_agg)
        direct_agg_agg_dist = calc_dist_from_aggregate(sushi_perms_2l[all_jpn_idx], direct_agg_agg)
        footrule_agg_agg_dist = calc_dist_from_aggregate(sushi_perms_2l[all_jpn_idx], footrule_agg_agg)
        
        
        print(samples_per, lehmer_agg_agg_dist, direct_agg_agg_dist, footrule_agg_agg_dist)
        lehmer_v_samples.append(lehmer_agg_agg_dist)
        direct_v_samples.append(direct_agg_agg_dist)
        footrule_v_samples.append(footrule_agg_agg_dist)
        
        #centralized aggregates
        lehmer_central = lehmer_max_consensus(central_perms)
        lehmer_central_dist = calc_dist_from_aggregate(sushi_perms_2l[all_jpn_idx], lehmer_central)

        direct_central = average_consensus(central_perms)
        direct_central_dist = calc_dist_from_aggregate(sushi_perms_2l[all_jpn_idx], direct_central)

        footrule_central = footrule_consensus(central_perms)
        footrule_central_dist = calc_dist_from_aggregate(sushi_perms_2l[all_jpn_idx], footrule_central)

        lehmer_v_central.append(lehmer_central_dist)
        direct_v_central.append(direct_central_dist)
        footrule_v_central.append(footrule_central_dist)
    
        
    fig, ax1 = plt.subplots(figsize=(8,6))
    
    samples_per_list = [int(x*87) for x in samples_frac]

    ax1.plot(samples_per_list, lehmer_v_samples, label='Lehmer coordinate majority', color='tab:blue')
    ax1.plot(samples_per_list, direct_v_samples, label='Borda coordinate quantization', color='tab:orange')
    ax1.plot(samples_per_list, footrule_v_samples, label='Footrule bipartite matching', color='tab:green')

    ax1.plot(samples_per_list, lehmer_v_central, color='tab:blue', linestyle='--')
    ax1.plot(samples_per_list, direct_v_central, color='tab:orange', linestyle='--')
    ax1.plot(samples_per_list, footrule_v_central, color='tab:green', linestyle='--')

    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    #ax2.set_xticklabels([int(x * num_clients) for x in ax1.get_xticks()])
    ax2.set_xticklabels([int(x/87 * 4981) for x in ax1.get_xticks()])
    
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax1.set_xlabel('Minimun number of samples at any client', fontsize=14)
    ax1.set_ylabel('Centralized objective', fontsize=14)
    ax2.set_xlabel('Total number of samples (with possible repetitions)', fontsize=14)
    ax1.legend(fontsize=14)
    ax1.set_title('Sushi preference', fontsize=14)

    plt.savefig('../figures/samples_sushi06.png', bbox_inches='tight')
    plt.close()
    #plt.show()
    return 0

if __name__ == "__main__":
    aggregate_communicate_sushi4()