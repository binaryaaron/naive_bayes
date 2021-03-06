"""
utils.py is file for helper methods
"""
import csv
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import metrics
except ImportError:
    raise ImportError("This program requires Numpy, sklearn, and Matplotlib")

__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


def read_file_dict(filename):
    """ Reads a one-column file and fills a dict with that data. Mostly for the
        vocabulary file and label file. Much faster than the iterable style
        Args:
            filename (str) : path to the file you want to open
        Returns:
            mydict: Dictionary of vocabulary
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        # dict comprehension over lines in file. Enumerate gives explicit i
        mydict = {i+1: line for i, line in enumerate(reader)}
    return mydict


def read_file_np(filename, delim=' ', ftype='int', filesize='1467345'):
    """ Reads a file  into a numpy structured array if ftype is overriden.
        Args:
            filename (str) : path to the file you want to open
            delim (str): a delimiter for the file
    """
    arr = np.loadtxt(filename, delimiter=delim, dtype=ftype)
    return(arr)


def get_words_from_doc(docid, model, vocab):
    """ Helper function to get the words represented from a document's bag of
    words model.
    Args:
        docid (int): document id in the BOW model.
        bow (nparr): bag of words model
        vocab (dict): dictionary for the model.
    Returns:
        List of words from that document.
    """
    # subsets the bow model into just the doc we want
    doc = model[model[:, 0] == docid]
    words = [vocab[doc[i][1]][0] for i in range(0, doc.shape[0])]
    return words


def get_class_from_bow(class_label, bow):
    """ Helper function to get the words represented from a document's bag of
    words model.
    Args:
        docid (int): document id in the BOW model.
        bow (nparr): bag of words model
        vocab (dict): dictionary for the model.
    Returns:
        view of matrix that we want.
    """
    # subsets the bow model into just the doc we want
    doc = bow[bow[:, 3] == class_label]
    return doc


def label_bow_with_class(label_dict, docid_vec):
    """Helper function to get a full class of labels from a bow matrix
    Args:
        label_dict (dict): dictionary of labels for the words
        docid_vec (np.array): array for the docids
    returns:
        np.array with filled labels
    """
    label_vec = np.zeros(docid_vec.shape, dtype='int8')
    for i in range(docid_vec.size):
        lab = docid_vec[i]
        label_vec[i] = label_dict[lab][0]
    return label_vec


def gen_agg_file(vocab, fileprefix, doc_dict):
    """Helper function to generate the nice version of the file we want. should
    only be used once
    """
    filename = fileprefix + '.data'

    print("Generated the aggregated file")
    docid = np.loadtxt(filename, dtype='int16', usecols=[0])
    wordid = np.loadtxt(filename, dtype='int32', usecols=[1])
    count = np.loadtxt(filename, dtype='int16', usecols=[2])
    print("Making label vector")
    labels = label_bow_with_class(doc_dict, docid)

    bow = np.column_stack((docid, wordid, count, labels))
    print("Saving file")
    np.savetxt(fileprefix + '.full', bow, fmt='%d')
    print("File saved")
    return bow


def read_mapfile(filename):
    """ Reads the file that maps word ids to actual words
    Args:
        filename: the .map filename
    Returns: dictionary of the map
    """
    with open('../data/train.map') as mapfile:
        reader = csv.reader(mapfile, delimiter=' ')
        label_map = {int(line[1]): line[0] for line in reader}
    return label_map


def estimate_priors(label_map, label_vec, num_classes=20):
    """populates a dictionary with the prior probabilities of each class
    Args:
        label_map (dict): class labels
        label_vec (np.array): vector of ids
    Returns: dictionary of prior probs for each class
    """
    priors = {}
    for i in range(1, num_classes+1):
        view = label_vec[label_vec == i]
        priors[label_map[i]] = view.size/label_vec.size
    return priors


def print_word_and_count(wordid, vocab, class_map, countsum_words):
    """Utility function to print the number of words in a class
    Args:
        wordid (int): word id in the dictionary
        vocab (dict): vocabulary dict
        class_map (dict): maps classes to values
        countsum_words: word list
    """
    print(vocab[wordid])
    for i in range(1, 20):
        print(class_map[i] + ': ' + str(countsum_words[wordid][i]))


def calc_gen_accuracy(predicted):
    """Reports the general accuracy of a predicted set of data
    Args:
        predicted (np.array): shape of features, 3 with labels, true,
                                predicted, and match
    return:
        tuple
    """
    num_right = np.sum(predicted[:, 2])
    num_wrong = predicted.shape[0] - num_right
    general_acc = num_right/predicted.shape[0]
    return(num_right, num_wrong, general_acc)


def report(predicted, labels, print_report=False,
           print_cm=True):
    """Light wrapper around sklearn metrics"""

    cr = metrics.classification_report(predicted[:, 0],
                                       predicted[:, 1],
                                       target_names=labels)
    acc = metrics.accuracy_score(predicted[:, 0],
                                 predicted[:, 1]
                                 )
    # print(acc)
    if print_report is True:
        print(cr)

    if print_cm is True:
        # Compute confusion matrix
        cm = metrics.confusion_matrix(predicted[:, 0], predicted[:, 1])

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / \
                        cm.sum(axis=1)[:, np.newaxis]
        # print('Normalized confusion matrix')
        plt.figure()
        plot_confusion_matrix(cm_normalized,
                              labels,
                              title='Normalized confusion matrix'
                              )
        plt.savefig('nb_confusion_matrix.pdf')
        plt.show()

    return (cr, acc)


def plot_confusion_matrix(cm, labels, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Example taken straight from scikit
    http://scikit-learn.org/dev/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def top_words(vectorizer, phat_w, class_labels, n=5, per_class=True,
              order='high'):
    """Prints features with the highest coefficient values, per class
    Originally seen on Scikit Learn for the per-class method, but I implemented
    the overall

    """
    words = vectorizer.get_feature_names()
    if per_class is True:
        for i, class_label in enumerate(class_labels):
            topwords_view = np.argsort(phat_w[i])[-n:]
            topwords = [words[j] + ',' for j in topwords_view]
            print("%s: %s" % (class_label, " ".join(topwords)))
        return
    else:
        # this gets the row vector of max arguments along classes
        view = np.argmax(phat_w, axis=0)
        # this slice takes from length-n -> length slice
        max_words = np.choose(view, phat_w)
        if order == 'high':
            maxes = np.argsort(max_words)[-n:]
        if order == 'low':
            maxes = np.argsort(max_words)[0:n]
        topwords = [words[j] for j in maxes]
        # for i, word in enumerate(topwords):
        # print("%d: %s" (i, word))
        print(topwords)


def get_word_count(cv, bow, word):
    """ Gets the count of a word in the bow model.
    """
    word_id = cv.vocabulary_[word]
    s = np.sum(bow[:, word_id].toarray())
    return (word, s)
