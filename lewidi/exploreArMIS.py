import json
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.optimize
from lewidi import *


params = [0, 1, 1, 1, .5]
bounds = ([-.5, -.5, 0, 0, 0], [1.5, 1.5, 1, 1, 1])
f = open('../../le-wi-di.github.io/data_post-competition/ArMIS_dataset/ArMIS_test.json')
analyze(f, params, bi_norm_optimizer, bounds, "ArMIS")
