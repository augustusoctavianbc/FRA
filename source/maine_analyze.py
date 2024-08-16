import numpy as np
import pandas as pd
import xlrd
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


data = pd.read_excel('../data/2018_June - Maine/Maine Democratic Primary for Governor CVR.xlsx')

for col_name in data.columns[3:]:
    #print(col_name)
    data[col_name].replace('skipped', 'none', inplace=True)
    data[col_name].replace('overvote', 'none', inplace=True)
    data[col_name].replace('undervote', 'none', inplace=True)
    
for col_name in data.columns[3:]:
    #print(col_name)
    data[col_name].replace('Mills, Janet T.', 'Mills', inplace=True)
    data[col_name].replace('Cote, Adam Roland', 'Cote', inplace=True)
    data[col_name].replace('Sweet, Elizabeth A.', 'Sweet', inplace=True)
    data[col_name].replace('Eves, Mark W.', 'Eves', inplace=True)
    data[col_name].replace('Dion, Mark N.', 'Dion', inplace=True)
    data[col_name].replace('Russell, Diane Marie', 'Russell', inplace=True)
    data[col_name].replace('Dion, Donna J.', 'Donna', inplace=True)
    data[col_name].replace('Write-in', 'Writein', inplace=True)
    


rank_default = ['Mills', 'Cote', 'Eves',  'Sweet', 'Dion', 'Russell', 'Donna', 'Writein']


data['ranks'] = data[data.columns[3:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

vote_types = data.ranks.unique()
vote_ranks = {}

perms = []
##cant do this at vote type level because we need to complete preferecences independently 

for vote_type in data['ranks']:
    vote_list = vote_type.split(',')
    vote_unique = []
    for vote in vote_list:
        if(vote != 'none'):
            if(vote not in vote_unique):
                vote_unique.append(vote)
                
    rank_list = [0]*len(rank_default)
    
    for itr, candi in enumerate(rank_default):
        try:
            rank_list[itr] = vote_unique.index(candi)+1
        except:
            continue
    
    
    vote_unique_len = len(vote_unique)
    if(vote_unique_len<8):
        vote_remain = [x for x in range(vote_unique_len+1,9)]
        np.random.shuffle(vote_remain)
        counter = 0
        for itr in range(len(rank_list)):
            if(rank_list[itr] == 0):
                rank_list[itr] = vote_remain[counter]
                counter+=1
    #print(rank_list, vote_unique)
    perms.append(rank_list)
    #break

perms = np.array(perms)


def aggregate_communicate_maine(perms):
    #number of samples increases in centralized version as well
    #distance at each iteration is calculated wrt all the samples
    
    #rows shuffled for randomized sample allocation
    np.random.shuffle(perms)
    
    #convert all rankings into Lehmer Codes
    n = len(perms)
    m = len(perms[0])
    
    #estimate phi by sampling some permutations 
    phi=0.6
    threshold = calc_threshold(phi,m,100000)
    
    num_clients = 50
    samples_per_list = [itr for itr in range(1,int(len(perms)/num_clients)+1,5)]
    samples_max = samples_per_list[-1]
    
    
    lehmer_v_samples = []
    direct_v_samples = []
    footrule_v_samples = []
    
    lehmer_v_central = []
    direct_v_central = []
    footrule_v_central = []
    
    #for real data and large permutations we set the numner of repetitions to 1
    #to keep the computational cost low
    repetitions = 1
    for reps in range(repetitions):
        lehmer_v_samples_reps = []
        direct_v_samples_reps = []
        footrule_v_samples_reps = []
        
        lehmer_v_central_reps = []
        direct_v_central_reps = []
        footrule_v_central_reps = []
        for samples_per in samples_per_list:
            central_perms = []
            
            client_lehmer_agg = []
            client_direct_agg = []
            client_footrule_agg = []
            for client in range(num_clients):
                
                client_perms = perms[client*samples_max:client*samples_max + samples_per, :]
                central_perms.extend(client_perms)
                
                
                client_lehmer_agg.append(lehmer_max_consensus(client_perms))
                client_direct_agg.append(bin_average_consensus(client_perms, threshold))
                client_footrule_agg.append(footrule_consensus(client_perms))
                
            lehmer_agg_agg = lehmer_max_consensus(client_lehmer_agg)
            direct_agg_agg = average_consensus(client_direct_agg)
            footrule_agg_agg = footrule_consensus(client_footrule_agg)
            
            
            lehmer_agg_agg_dist = calc_dist_from_aggregate(perms, lehmer_agg_agg)
            direct_agg_agg_dist = calc_dist_from_aggregate(perms, direct_agg_agg)
            footrule_agg_agg_dist = calc_dist_from_aggregate(perms, footrule_agg_agg)
            
    
            lehmer_v_samples_reps.append(lehmer_agg_agg_dist)
            direct_v_samples_reps.append(direct_agg_agg_dist)
            footrule_v_samples_reps.append(footrule_agg_agg_dist)
            
            #centralized aggregates
            lehmer_central = lehmer_max_consensus(central_perms)
            lehmer_central_dist = calc_dist_from_aggregate(perms, lehmer_central)
            
            direct_central = average_consensus(central_perms)
            direct_central_dist = calc_dist_from_aggregate(perms, direct_central)
            
            footrule_central = footrule_consensus(central_perms)
            footrule_central_dist = calc_dist_from_aggregate(perms, footrule_central)
            
            
            lehmer_v_central_reps.append(lehmer_central_dist)
            direct_v_central_reps.append(direct_central_dist)
            footrule_v_central_reps.append(footrule_central_dist)
            print(samples_per, time.time()-t1, lehmer_agg_agg, lehmer_central)
        
        lehmer_v_samples.append(lehmer_v_samples_reps)
        direct_v_samples.append(direct_v_samples_reps)
        footrule_v_samples.append(footrule_v_samples_reps)
        
        lehmer_v_central.append(lehmer_v_central_reps)
        direct_v_central.append(direct_v_central_reps)
        footrule_v_central.append(footrule_v_central_reps)
        
    
    lehmer_v_samples = np.array(lehmer_v_samples)
    direct_v_samples = np.array(direct_v_samples)
    footrule_v_samples = np.array(footrule_v_samples)
    
    lehmer_v_central = np.array(lehmer_v_central)
    direct_v_central = np.array(direct_v_central)
    footrule_v_central = np.array(footrule_v_central)
    
    lehmer_v_samples = np.mean(lehmer_v_samples, axis=0)
    direct_v_samples = np.mean(direct_v_samples, axis=0)
    footrule_v_samples = np.mean(footrule_v_samples, axis=0)
    
    lehmer_v_central = np.mean(lehmer_v_central, axis=0)
    direct_v_central = np.mean(direct_v_central, axis=0)
    footrule_v_central = np.mean(footrule_v_central, axis=0)
    
        
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
    ax1.set_title('Maine Primary', fontsize=14)

    plt.savefig('../figures/samples_mainegov06_50_rd4.png', bbox_inches='tight')
    plt.close()
    print(time.time()-t1)
    return 0

if __name__ == "__main__":
    aggregate_communicate_maine(perms)