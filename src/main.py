#!/usr/bin/env python
"""Naive Bayes Classifier

Usage:
    main.py    TRAIN TEST VOCAB [--ipython | -h]

Arguments:
    train        the training data prefix
    test         the testing data prefix
    vocab        the vocabulary file
Options:
    -h, --help     Show this screen.
    --ipython      using ipython notebook and we to forgo saving figs
    --vocab        vocabulary file
"""
from docopt import docopt

"""
main.py is the main entry point for the classifier.
"""
import csv

# graph tool
try:
    import numpy as np
except ImportError:
    raise ImportError("This program requires Numpy and Matplotlib")


__author__ = "Aaron Gonzales"
__copyright__ = "GPL"
__license__ = "GPL"
__maintainer__ = "Aaron Gonzales"
__email__ = "agonzales@cs.unm.edu"


def read_file_dict(filename):
    """ Reads a one-column file and fills a dict with that data. Mostly for the vocabulary
        file and label file. Much faster than the iterable style
        Args:
            filename (str) : path to the file you want to open
        Returns:
            mydict: Dictionary of vocabulary
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        # dict comprehension over lines in file. Enumerate gives explicit i
        mydict = {i+1:line for i,line in enumerate(reader)}
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
    doc = model[model[:,0] == docid]
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
    doc = bow[bow[:,3] == class_label]
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
    with open('../data/train.map') as mapfile:
        reader = csv.reader(mapfile, delimiter=' ')
        #for line in reader:
         #   print(line)
        label_map = {int(line[1]):line[0] for line in reader}
    return label_map


def main(_args):
    """ Drives the program."""
    if _args["--ipython"]:
        print("Ipython session selected; no saving of figures will happen")
    print("--------Training File: {computer}".format(computer=_args["TRAIN"]))
    print("--------Testing File: {computer}".format(computer=_args["TEST"]))

    train_data = read_file_np(_args['TRAIN'] + '.data',
                              ftype='int16, int32, int32')
    # print("Read " + train_data.shape + " from " + _args['TRAIN'])
    train_vocab = read_vocab(_args['VOCAB'])
    train_doc_labs = read_file_dict(_args['TRAIN'] + '.label')
    label_map = read_mapfile(_args['TRAIN'] + '.map')

    # if good file exists:
        # train_data = read_file_np(_args['TRAIN'] + '.data',
                                  # ftype='int16, int32, int32, int8')
        # load that file
    # else: 
        # generate that good file



if __name__ == "__main__":
    main(docopt(__doc__))
