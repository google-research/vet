from matplotlib import pyplot as plt
import json
from scipy.stats import norm
import numpy as np
import scipy.optimize
import pdb

def norm_optimizer(x, *args):
    """A cover function for fitting the normal distribution.

    Args:
        x: the data to fit.
        args: the two paramaters (expected value and standard
            deviation) of the normal distribution being fitted.
    
    Returns:
        The probabilit(ies) associated with the given input.
    """
    m1, s1 = args
    ret = scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
    return ret

def bi_norm_optimizer(x, *args):
    """A cover function for fitting the binormal distribution.

    Args:
        x: the data to fit.
        args: the two paramaters (expected value and standard
            deviation) of the normal distribution being fitted.
    
    Returns:
        The probabilit(ies) associated with the given input.
    """
    m1, m2, s1, s2, k = args
    ret = k*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
    ret += (1-k)*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
    return ret

def bounded_sample(mean, std, range, size):
    """Sample from [0,1] via a normal distribution.

     Vars:
        mean: The mean of the normal distribution to sample from.
        std: The standard deviation of the distribution to sample from.
		range: an tuple containing upper and lower bounds within which
			the sample must be contained.
		size: the size of the sample (i.e., number of items in the sample).
    Returns:
        A size "size" sample, as a list.
    """
    sample_items = []
    lower, upper = range
    for _ in range(size):
        sample_of_one = scipy.stats.norm.rvs(loc=mean, scale=s, size=1)
        while ((n < lower) or (n > upper)):
            sample_of_one = scipy.stats.norm.rvs(loc=mean, scale=s, size=1)
        sample_items.append(sample_of_one[0])
    return sample_items 

def get_dists(vals):
	"""Return the binned distribution of vals.

	Note: as a side-effect, this method creats a matplotlib buffer
		that can be displayed.

	Args:
		vals: a list of values

	Returns:
		A list the name size a vals, containing the frequences of 
		each value in vals.
	"""
	counts, bins, _ = plt.hist(vals)
	y = np.searchsorted(bins, vals, side="right")
	y = [int(min(i-1, len(counts)-1)) for i in y]
	return [counts[i]/len(vals) for i in y]

def analyze(f, params, generator, bounds, file_name):
	train = json.load(f)
	annotations = [ 
		i['annotations'] for k,i in train.items()
	]
	annotations = [[int(i) for i in a.split(',')] for a in annotations]
	mus = sorted([np.mean(i) for i in annotations])
	stdevs = sorted([np.std(i) for i in annotations])
	analyze2(mus, stdevs, params, generator, bounds, file_name)

def analyze2(mus, stdevs, params, generator, bounds, file_name):
	mu_counts = get_dists(mus)
	plt.show()

	plt.plot(mus, mu_counts, 'o')
	#plt.show()
	mu_fitted_params,_ = scipy.optimize.curve_fit(generator, mus, mu_counts, p0=params, bounds=bounds)
	if len(mu_fitted_params) == 2:
		print("Parameters for mean: m: %f, s: %f" % tuple(mu_fitted_params))
	else:
		print("Parameters for mean: m1: %f, m2: %f, s1: %f, s2: %f, k: %f" % tuple(mu_fitted_params))
	xx = np.linspace(np.min(mus), np.max(mus), 1000)
	plt.plot(xx, generator(xx, *mu_fitted_params))
	plt.show()
	plt.savefig(f"{file_name}_mean.pdf")


	plt.clf()
	stdev_counts = get_dists(stdevs)
	plt.show()

	plt.plot(stdevs, stdev_counts, 'o')
	#plt.show()
	std_fitted_params,_ = scipy.optimize.curve_fit(generator, stdevs, stdev_counts, p0=params, bounds=bounds)
	if len(std_fitted_params) == 2:
		print("Parameters for std: m: %f, s: %f" % tuple(std_fitted_params))
	else:
		print("Parameters for std: m1: %f, m2: %f, s1: %f, s2: %f, k: %f" % tuple(std_fitted_params))
	xx = np.linspace(np.min(stdevs), np.max(stdevs), 1000)
	plt.plot(xx, generator(xx, *std_fitted_params))
	plt.show()
	plt.savefig(f"{file_name}_std.pdf")


