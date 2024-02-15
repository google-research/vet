from matplotlib import pyplot as plt
import json
from scipy.stats import norm
import scipy.optimize
import numpy as np
from lewidi import *

params = [0, 1]
bounds = ([-.5,0],[1.5,1])
f = open('../../le-wi-di.github.io/data_post-competition/ConvAbuse_dataset/ConvAbuse_test.json')
analyze(f, params, norm_optimizer, bounds, "ConvAbuse")
