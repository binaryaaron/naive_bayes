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


def calc_priors(train_data):
    """calculates the priors of a class"""
    w_given_class = phat_word_est(train_data)


def phat_word_est(train_data, laplacian=False):
    """Gets P(c|w) for all words in the dictionary.
    Args:
        train_data (numpy.array): training set
        laplacian (bool): denote if laplacian is wanted or not
    Returns: numpy array (float16) of all words and estimates
    """
    word_tots = []
    countsum_words = []
    # testing on small set
    # TODO change this
    for i in range(1, 10000):
        count = train_data[train_data[:, 1] == i]
        tot_word = count.sum(axis=0)[2]
        word_tots.append(tot_word)
        # print(count)
        count_tmp = []
        for i in range(1, 21):
            # return view over single classes
            class_view = count[count[:, 3] == i]
            # gets scalar sum for the counts in the view
            # and returns just that scalar value; Numpy returns an array
            class_sum = class_view.sum(axis=0)[2]
            count_tmp.append(class_sum/tot_word)
        countsum_words.append(count_tmp)
    phat_words = np.array(countsum_words, dtype='float16')
    countsum_words = []
    word_tots = []
    return phat_words
