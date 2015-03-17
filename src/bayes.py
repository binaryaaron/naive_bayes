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


def phat_word_est(bow_train, class_labels, nclasses=20, alpha=None):
    """Gets P(c|w) for all words in the dictionary.
    Args:
        bow_train (scipy.sparse): training set, assumes bag of words
        class_labels(np.array): list of class label names (scikitlearn data.target_names)
        class_names(list): array of class labels for each document
        laplacian (bool): denote if laplacian is wanted or not
        nclasses (int): number of classes in the set
    Returns: tuple with (numpy array of summed words, numpy array of priors)
    """
    vocab_n = bow_train.shape[0]
    if alpha is not None:
        # this is done for easy of the for loop's calculation.
        _alpha = alpha+1
        denominator_p = alpha*nclasses
        print('estimating params with Laplacian Smoothing:')
        print('\t alpha = %d' % alpha)
        print('\t denominator = %d' % denominator_p)
    else:
        # beta is 1/(vocab size)
        beta = 1/vocab_n
        _alpha = 1 + beta
        denominator_p = beta * vocab_n
        print('estimating params with Dirchlet Prior:')
        print('\t vocabsize = %f' % vocab_n)
        print('\t beta = %f' % beta)
        print('\t alpha = %f' % _alpha)
        print('\t denominator = %f' % denominator_p)

    # gives the total number of words for a column vector in bow model
    word_sums = bow_train.sum(axis=0)
    # preallocate array, 20 row, word dic length
    phat_words = np.zeros((nclasses, bow_train.shape[1]), dtype='float64')

    # create sum vectors and assign them to the correct spot
    # may need to fiddle with smoothing params
    for i in range(nclasses):
        c_mask = class_labels == i
        c_sum = bow_train[c_mask].sum(axis=0)
        c_sum = c_sum + (_alpha - 1)
        # print(c_sum/word_sums)
        phat_words[i] = (c_sum / (word_sums + denominator_p))
    # return (word_sums, phat_words)
    return phat_words
