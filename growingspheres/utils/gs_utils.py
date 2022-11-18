#!/usr/bin/env python
# -*- coding: utf-8 -*-

import n_sphere
import math
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances

    
def get_distances(x1, x2, metrics=None):
    x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
    euclidean = pairwise_distances(x1, x2)[0][0]
    same_coordinates = sum((x1 == x2)[0])
    
    #pearson = pearsonr(x1, x2)[0]
    kendall = kendalltau(x1, x2)
    out_dict = {'euclidean': euclidean,
                'sparsity': x1.shape[1] - same_coordinates,
                'kendall': kendall
               }
    return out_dict   

def generate_inside_ball(center, segment, n):
    """
    generate n points in the space between two balls centered in center and 
    of radius segment[0] and segment[1]
    """
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    z = np.random.normal(0, 1, (n, d))
    r = np.random.uniform(segment[0], segment[1], n)
    #u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    #r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
    z = z + center
    return z
    
# tentativo di coordinate sferiche
#def generate_inside_ball(center, segment, n):
#    d = center.shape[1]
#    print('----')
#    print('ri:',segment[0])
#    print('rf:',segment[1])
#    r = np.random.uniform(segment[0], segment[1], (n,1))
#    angles = np.random.uniform(0,math.pi,(n, d-2))
#    phi_n = np.random.uniform(0,2*math.pi,(n, 1))
#    s = np.hstack((r,angles,phi_n))
#    z = n_sphere.convert_rectangular(s)
#    z += center
#    return z

    
