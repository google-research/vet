from matplotlib import pyplot as plt
import json
from scipy.stats import norm
import numpy as np
import scipy.optimize
from lewidi import *
#import pdb

 

params = [.5, 1]
bounds = ([-.5,0],[1.5,1])
f = open('../../le-wi-di.github.io/data_post-competition/HS-Brexit_dataset/HS-Brexit_test.json')
analyze(f, params, norm_optimizer, bounds, "HS-Brexit")
