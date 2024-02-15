import pandas as pd
from lewidi import *

amazon = pd.read_csv("../../product_ratings.csv")

amazon = amazon.set_index("product_id")

amazon = (amazon-1)/5
mus = list(amazon.mean(axis=0))
stdevs = list(amazon.std(axis=0))

params = [.5,.5]
bounds = ([-.5,0],[1.5,1])
analyze2(mus, stdevs, params, norm_optimizer, bounds, "Amazon")
