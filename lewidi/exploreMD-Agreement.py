from matplotlib import pyplot as plt
import pdb
import json
from scipy.stats import norm
import numpy as np
import scipy.optimize
from lewidi import *

params = [0, 1]
bounds = ([-.5,0],[1.5,1])
f = open('../../le-wi-di.github.io/data_post-competition/MD-Agreement_dataset/MD-Agreement_test.json')
analyze(f, params, norm_optimizer, bounds)
