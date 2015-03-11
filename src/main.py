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
import utils

__author__ = "Aaron Gonzales"
__copyright__ = "GPL"
__license__ = "GPL"
__maintainer__ = "Aaron Gonzales"
__email__ = "agonzales@cs.unm.edu"


def main(_args):
    """ Drives the program."""
    if _args["--ipython"]:
        print("Ipython session selected; no saving of figures will happen")
    print("--------Training File: {computer}".format(computer=_args["TRAIN"]))
    print("--------Testing File: {computer}".format(computer=_args["TEST"]))

    # train_data = utils.read_file_np(_args['TRAIN'] + '.data',
                                    ftype='int16, int32, int32')
    # print("Read " + train_data.shape + " from " + _args['TRAIN'])
    # train_vocab = utils.read_file_dict(_args['VOCAB'])
    # train_doc_labs = utils.read_file_dict(_args['TRAIN'] + '.label')
    # label_map = utils.read_mapfile(_args['TRAIN'] + '.map')
    train_data = utils.read_file_np(_args['TRAIN'] + '.full', ftype = 'uint16')


if __name__ == "__main__":
    main(docopt(__doc__))
