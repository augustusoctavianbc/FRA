import math
import numpy as np
import mallows_kendall as mk
import random
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from collections import Counter
#import pyflagr
import xlrd
import matplotlib.pyplot as plt
import time


def generate_mallows_ranking(n,m,phi, sigma_0):
    #m and n have opposite roles in the mallows_kendall library
    #mallow_kendall library also works with rankings starting from zero
    #adjust the ranking here to shhit to 1 base ranking
    #adjust it back after generating
    sigma_0 = [itr-1 for itr in sigma_0]
    perms = mk.sample(m=n, n=m, phi=phi, s0=sigma_0)
    perms = np.array(perms)
    #print(perms)
    perms = perms+1
    #print(perms)
    return perms


def invert_perm(p):
    '''
    Convert between two-line and one-line notation
    Parameters
    ----------
    p : permutation

    Returns
    -------
    p_inv : permutation

    '''
    
    p_inv = np.zeros(len(p))
    for itr in range(len(p)):
        p_inv[int(p[itr])-1] = itr+1
    return p_inv

def average_consensus(perms):
    '''

    Parameters
    ----------
    perms : list of permutations
        The permuations are in two-line notation;
        sigma(x) = i indicates that element x has rank i;

    Returns
    -------
    ranks : permutation
        aggregate of permutations;
        Implements Borda with random tie breaks;
        Coordinate-wise averaging

    '''
    #n = len(perms)
    #m = len(perms[0])
    #avg_rank = (sum(perms)+[random.uniform(10**(-3), 10**(-4)) for itr in range(m)])/n
    avg_rank = sum(perms)
    ranks = rankdata(avg_rank, method='ordinal')
    return ranks

def average_inv_consensus(perms):
    '''
    
    Parameters
    ----------
    perms : list of permutations
        The permutations are in one-line notation;
        sigma(x) = i indicates that element i is at rank x;
    Returns
    -------
    p : permutation
        Implements Borda with random tie breaks;
        First invert the permutation and then do coordinate-wise averaging

    '''
    perms_inv = []
    for p in perms:
        perms_inv.append(invert_perm(p))
        
    avg_rank = sum(perms_inv)
    ranks = rankdata(avg_rank, method='ordinal')
    
    p = invert_perm(ranks)    
    return p


def graph_weight_consensus(perms):
    '''
    

    Parameters
    ----------
    perms : list of permutations
        The permutations are in two-line notation

    Returns
    -------
    ranks : permutation
        Aggregate permutations based on indegree of pairwise comparison;
        5-approximation

    '''
    n = len(perms)
    m = len(perms[0])
    indegree = np.zeros(m)
    #print(m, n)
    for itrx in range(m):
        for itry in range(m):
            if(itrx == itry):
                continue
            for itrz in range(n):
                if(perms[itrz][itrx] < perms[itrz][itry]):
                    indegree[itrx]-=1
    #rankdata gives smallest numerical rank to smallest score;
    #smallest score in this implementation would be for highest indegree
    ranks = rankdata(indegree, method='ordinal')
    return ranks


def footrule_consensus(perms):
    '''
    

    Parameters
    ----------
    perms : list of permutations
        The permutations are in two-line notation

    Returns
    -------
    ranks : permutation
        Aggregate permutations based spearman footrule bipartite matching;
        3-approximation

    '''
    n = len(perms)
    m = len(perms[0])
    bigraph = np.ones((m,m))
    for itrx in range(m):
        for itry in range(m):
            for itrz in range(n):
                bigraph[itrx][itry]+= np.absolute(perms[itrz][itrx] - (itry+1))
        
    #print(bigraph)
    bigraph_csr = csr_matrix(bigraph)
    ranks = min_weight_full_bipartite_matching(bigraph_csr)[1] + 1
    return ranks

def perm_2_lehmer(p):
    '''

    Parameters
    ----------
    p: permutation 

    Returns
    -------
    l : lehmer code

    '''
    l = np.zeros(len(p))
    for itrx in range(len(p)):
        for itry in range(itrx):
            if(p[itry]>p[itrx]):
                l[itrx]+=1
    return l

def lehmer_2_perm(l):
    '''

    Parameters
    ----------
    l : lehmer code

    Returns
    -------
    p: permutation 
    

    '''
    remain = [itr for itr in range(len(l), 0, -1)]
    p = np.zeros(len(l))
    for itrx in range(len(l)-1, -1, -1):
        p[itrx] = remain[int(l[itrx])]
        remain.remove(p[itrx])
    return p



def lehmer_max_consensus(perms):
    '''

    Parameters
    ----------
    lehmer_perms: permutations to aggregate

    Returns
    -------
    ranks : aggregate based on lehmer codes described in the paper

    '''
    n = len(perms)
    m = len(perms[0])
    lehmer_perms = []
    for p in perms:
        lehmer_perms.append(perm_2_lehmer(p))
    lehmer_perms = np.array(lehmer_perms)
    max_lehmer = []
    for itr in range(m):
        ctr = Counter(lehmer_perms[:,itr])
        hist = ctr.most_common(itr+1)
        r,_ = hist[0]
        max_lehmer.append(min(itr, r))
    #ranks = [r+1 for r in ranks]
    ranks = lehmer_2_perm(max_lehmer)
    return ranks

def lehmer_max_consensus_l(lehmer_perms):
    '''

    Parameters
    ----------
    lehmer_perms: lehmer code of permutations to aggregate

    Returns
    -------
    ranks : aggregate based on lehmer codes described in the paper

    '''
    #in this version of the function, we input the lehmer code itself
    #it helps to speed up the code by ammortizing the cost of perm to lehmer
    n = len(lehmer_perms)
    m = len(lehmer_perms[0])
    
    max_lehmer = []
    for itr in range(m):
        ctr = Counter(lehmer_perms[:,itr])
        hist = ctr.most_common(itr+1)
        r,_ = hist[0]
        max_lehmer.append(min(itr, r))
    #ranks = [r+1 for r in ranks]
    ranks = lehmer_2_perm(max_lehmer)
    return ranks

def calc_dist_from_aggregate(perms, agg_perm):
    '''

    Parameters
    ----------
    perms: permutations to calculate distance from
    
    agg_perm: aggregate or centroid permutation 

    Returns
    -------
    dist : normalized average Kemeney distance

    '''
    n = len(perms)
    m = len(perms[0])
    dist = 0
    for perm in perms:
        dist+=mk.distance(perm, agg_perm)
    
    return (dist/(n*m))


def calc_threshold(phi=0.1, m=10, n=10000):
    '''

    Parameters
    ----------
    phi : scaling factor (0,1)
        
    m : length of permutations
    
    n: number of samples to generate

    Returns
    -------
    thresh : bin threshold for quantized Borda

    '''
    
    sigma_0 = np.array(range(0,m,1))
    perms = mk.sample(m=n, n=m, phi=phi, s0=sigma_0)
    perms = np.array(perms)
    perms = perms+1
    #print(np.mean(perms, axis=0))
    avg_perm = np.mean(perms, axis=0)
    thresh = [np.mean((avg_perm[i],avg_perm[i+1])) for i in range(m-1)]
    #print(thresh)
    return thresh
    
def bin_average_consensus(perms, threshold):
    '''

    Parameters
    ----------
    perms : list of permutations
        The permutations are in two-line notation
        
    threshold : list of thresholds as described in the paper

    Returns
    -------
    ranks : permutation
        Aggregate permutations based quantized Borda

    '''
    avg_perm = np.mean(perms, axis=0)
    
    ranks = np.zeros(len(avg_perm))
    
    for itr in range(len(avg_perm)):
        flag = True
        for t in range(len(threshold)):
            if(avg_perm[itr]<threshold[t]):
                ranks[itr] = t+1
                flag = False
                break
        if flag:
            ranks[itr] = len(threshold)+1
    
    return ranks

def estimate_phi(perms):
    '''
    

    Parameters
    ----------
    perms : list of permutations
        The permutations are in two-line notation

    Returns
    -------
    best_phi : optimal value of phi

    '''
    m=len(perms[0])
    phi_list = [x/10 for x in range(1,10,1)]
    min_error = 100000
    best_phi = 0.1
    for phi in phi_list:
        
        threshold = calc_threshold(phi,m,10000)
        bin_agg = bin_average_consensus(perms, threshold)
        bin_agg_dist = calc_dist_from_aggregate(perms, bin_agg)
        print(phi, bin_agg_dist)
        if(bin_agg_dist<min_error):
            min_error = bin_agg_dist
            best_phi = phi
    return best_phi
