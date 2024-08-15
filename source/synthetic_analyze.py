import numpy as np
import pandas as pd
from scipy.stats import rankdata
import mallows_kendall as mk
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
from rankutils import generate_mallows_ranking

import matplotlib.pyplot as plt
import time

t1 = time.time()
np.random.seed(42)




def aggregate_communicate_n1(sigma_0, phi):
    #number of clients is fixed while the number of samples increases
    #number of samples increases in centralized version as well
    t1 = time.time()
    m = len(sigma_0)
    print(m,phi,sigma_0)
    
    threshold = calc_threshold(phi,m,100000)
    
    num_clients = 10
    samples_per_list = [itr for itr in range(1,51,1)]
    
    
    lehmer_v_samples = []
    direct_v_samples = []
    
    
    lehmer_v_central = []
    direct_v_central = []
    
    repetitions = 100
    for reps in range(repetitions):
        
        lehmer_v_samples_reps = []
        direct_v_samples_reps = []
        
        
        lehmer_v_central_reps = []
        direct_v_central_reps = []
        
        for samples_per in samples_per_list:
            central_perms = []
            
            client_lehmer_agg = []
            client_direct_agg = []
            
            for client in range(num_clients):
                #do we have the same sigma_0 at each client? yes
                #sigma_0 = np.array(range(1,m+1,1))
                
                #gnenerate perms at each client
                client_perms = generate_mallows_ranking(n=samples_per,m=m,phi=phi, sigma_0=sigma_0)
                #save the perms for evaluating centralized performace
                central_perms.extend(client_perms)
                
                client_lehmer_agg.append(lehmer_max_consensus(client_perms))
                client_direct_agg.append(bin_average_consensus(client_perms, threshold))
                
                
            lehmer_agg_agg = lehmer_max_consensus(client_lehmer_agg)
            direct_agg_agg = average_consensus(client_direct_agg)
            
            
            lehmer_agg_agg_dist = mk.distance(lehmer_agg_agg, sigma_0)
            direct_agg_agg_dist = mk.distance(direct_agg_agg, sigma_0)
            
            
            lehmer_v_samples_reps.append(lehmer_agg_agg_dist)
            direct_v_samples_reps.append(direct_agg_agg_dist)
            
            
            #centralized aggregates
            lehmer_central = lehmer_max_consensus(central_perms)
            lehmer_central_dist = mk.distance(lehmer_central, sigma_0)
            
            direct_central = average_consensus(central_perms)
            direct_central_dist = mk.distance(direct_central, sigma_0)
            
            
            lehmer_v_central_reps.append(lehmer_central_dist)
            direct_v_central_reps.append(direct_central_dist)
        
        lehmer_v_samples.append(lehmer_v_samples_reps)
        direct_v_samples.append(direct_v_samples_reps)
        
        lehmer_v_central.append(lehmer_v_central_reps)
        direct_v_central.append(direct_v_central_reps)
    
    lehmer_v_samples = np.array(lehmer_v_samples)
    direct_v_samples = np.array(direct_v_samples)
    
    lehmer_v_central = np.array(lehmer_v_central)
    direct_v_central = np.array(direct_v_central)
    
    lehmer_v_samples = np.mean(lehmer_v_samples, axis=0)
    direct_v_samples = np.mean(direct_v_samples, axis=0)
    
    lehmer_v_central = np.mean(lehmer_v_central, axis=0)
    direct_v_central = np.mean(direct_v_central, axis=0)
        
    sigma_string = ''
    for itr in sigma_0:
        sigma_string = sigma_string+str(itr)+','
        
    sigma_string = sigma_string[:-1]
        
    fig, ax1 = plt.subplots(figsize=(8,6))

    ax1.plot(samples_per_list, lehmer_v_samples, label='Lehmer coordinate majority', color='tab:blue')
    ax1.plot(samples_per_list, direct_v_samples, label='Borda coordinate quantization', color='tab:orange')

    ax1.plot(samples_per_list, lehmer_v_central, color='tab:blue', linestyle='--')
    ax1.plot(samples_per_list, direct_v_central, color='tab:orange', linestyle='--')
    
    

    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([int(x * num_clients) for x in ax1.get_xticks()])
    
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax1.set_xlabel('Samples at each client', fontsize=14)
    ax1.set_ylabel(r'Kendall $\tau$ distance', fontsize=14)
    ax2.set_xlabel('Total number of samples (with possible repetitions)', fontsize=14)
    ax1.legend(fontsize=14)
    ax1.set_title(r'$\sigma_0$: ['+sigma_string+r'] $\phi$: '+str(phi), fontsize=14)

    sigma_string = ''
    for itr in sigma_0:
        sigma_string = sigma_string+str(itr)
        
    plt.savefig('../figures/samples_sigma0'+sigma_string+'phi'+str(phi)+'m'+str(m)+'.png', bbox_inches='tight')
    plt.close()
    print(time.time()-t1)
    return 0


def aggregate_communicate_n2(sigma_0, phi):
    #number of clients increases while the number samples at each client is fixed
    #number of samples increases in centralized version as well
    t1 = time.time()
    m = len(sigma_0)
    print(m,phi,sigma_0)
    
    threshold = calc_threshold(phi,m,100000)
    
    num_clients_list = [itr for itr in range(1,51,1)]
    
    samples_per = 10
    
    
    lehmer_v_samples = []
    direct_v_samples = []
    
    lehmer_v_central = []
    direct_v_central = []
    
    repetitions = 100
    for reps in range(repetitions):
        lehmer_v_samples_reps = []
        direct_v_samples_reps = []
        
        lehmer_v_central_reps = []
        direct_v_central_reps = []
        
        for num_clients in num_clients_list:
            central_perms = []
            
            client_lehmer_agg = []
            client_direct_agg = []
            
            for client in range(num_clients):
                
                #generate perms at each client and save them to evaluate centralized performance
                client_perms = generate_mallows_ranking(n=samples_per,m=m,phi=phi, sigma_0=sigma_0)
                central_perms.extend(client_perms)
                
                client_lehmer_agg.append(lehmer_max_consensus(client_perms))
                client_direct_agg.append(bin_average_consensus(client_perms, threshold))
                
                
            lehmer_agg_agg = lehmer_max_consensus(client_lehmer_agg)
            direct_agg_agg = average_consensus(client_direct_agg)
            
            
            lehmer_agg_agg_dist = mk.distance(lehmer_agg_agg, sigma_0)
            direct_agg_agg_dist = mk.distance(direct_agg_agg, sigma_0)
            
            
            lehmer_v_samples_reps.append(lehmer_agg_agg_dist)
            direct_v_samples_reps.append(direct_agg_agg_dist)
            
            
            #centralized aggregates
            lehmer_central = lehmer_max_consensus(central_perms)
            lehmer_central_dist = mk.distance(lehmer_central, sigma_0)
            
            direct_central = average_consensus(central_perms)
            direct_central_dist = mk.distance(direct_central, sigma_0)
            
            
            lehmer_v_central_reps.append(lehmer_central_dist)
            direct_v_central_reps.append(direct_central_dist)
        
        lehmer_v_samples.append(lehmer_v_samples_reps)
        direct_v_samples.append(direct_v_samples_reps)
        
        lehmer_v_central.append(lehmer_v_central_reps)
        direct_v_central.append(direct_v_central_reps)
    
    lehmer_v_samples = np.array(lehmer_v_samples)
    direct_v_samples = np.array(direct_v_samples)
    
    lehmer_v_central = np.array(lehmer_v_central)
    direct_v_central = np.array(direct_v_central)
    
    lehmer_v_samples = np.mean(lehmer_v_samples, axis=0)
    direct_v_samples = np.mean(direct_v_samples, axis=0)
    
    lehmer_v_central = np.mean(lehmer_v_central, axis=0)
    direct_v_central = np.mean(direct_v_central, axis=0)
        
    
    sigma_string = ''
    for itr in sigma_0:
        sigma_string = sigma_string+str(itr)+','
        
    sigma_string = sigma_string[:-1]
        
    fig, ax1 = plt.subplots(figsize=(8,6))

    ax1.plot(num_clients_list, lehmer_v_samples, label='Lehmer coordinate majority', color='tab:blue')
    ax1.plot(num_clients_list, direct_v_samples, label='Borda coordinate quantization', color='tab:orange')

    ax1.plot(num_clients_list, lehmer_v_central, color='tab:blue', linestyle='--')
    ax1.plot(num_clients_list, direct_v_central, color='tab:orange', linestyle='--')

    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([int(x * samples_per) for x in ax1.get_xticks()])

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    ax1.set_xlabel('Number of clients', fontsize=14)
    ax1.set_ylabel(r'Kendall $\tau$ distance', fontsize=14)
    ax2.set_xlabel('Total number of samples (with possible repetitions)', fontsize=14)
    ax1.legend(fontsize=14)
    ax1.set_title(r'$\sigma_0$: ['+sigma_string+r'] $\phi$: '+str(phi), fontsize=14)

    
    sigma_string = ''
    for itr in sigma_0:
        sigma_string = sigma_string+str(itr)
        
    plt.savefig('../figures/clients_sigma0'+sigma_string+'phi'+str(phi)+'m'+str(m)+'.png', bbox_inches='tight')
    plt.close()
    print(time.time()-t1)
    return 0

def main():
    
    sigma_0 = np.array(range(1,11,1))
    #sigma_0 = np.array(range(1,21,1))
    
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.2)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.2)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.4)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.4)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.6)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.6)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.1)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.1)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.3)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.3)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.5)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.5)
    

    #sigma_0 = np.array([1,2,4,3,5,7,6,8,9,10,11,12,13,14,16,15,17,18,20,19])
    sigma_0 = np.array([1,2,4,3,5,7,6,8,9,10])
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.2)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.2)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.4)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.4)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.6)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.6)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.1)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.1)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.3)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.3)
    aggregate_communicate_n1(sigma_0=sigma_0, phi=0.5)
    aggregate_communicate_n2(sigma_0=sigma_0, phi=0.5)
    
if __name__ == "__main__":
    main()
