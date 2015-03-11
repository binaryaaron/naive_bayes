"""
bayes.py handles implementation of the meaty parts of the classifier
"""
try:
    import numpy as np
except ImportError:
    raise ImportError("This program requires Numpy")

__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


def estimate(a, x, xga):
    """ uses bayes's theorem to estimate a posterior probability
    Args:
        a (float) : probability of a
        x (float) : probability of b
        xga : probabilty of x|a ("x given a")
        Returns:
            float: probability of a | x ("a given x")
    """
    # testing with a neg num
    agx = -1.0


def calc_priors(features):
    """calculates the priors of a class"""





