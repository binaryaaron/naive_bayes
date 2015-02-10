#!/usr/bin/env python
"""
main.py is the main entry point for the classifier.
"""
import sys
import argparse
import os
import csv
from pprint import pprint

# graph tool
try:
  import numpy as nx
  import matplotlib.pyplot as plt
except ImportError:
  raise ImportError("This program requires Numpy and Matplotlib")


__author__ = "Aaron Gonzales"
__copyright__ = "GPL"
__license__ = "GPL"
__maintainer__ = "Aaron Gonzales"
__email__ = "agonzales@cs.unm.edu"


def read_file(data_list, filename):
  """ Reads a file with DNA promoter data
      and fills a list with that data.
      Args:
        data_list (list) : the empty list you want to put data into
        filename (str) : path to the file you want to open
  """
  with open(filename, 'rb') as f:
    reader = csv.reader(f, delimiter=' ')
    for line in f:
      # splits the line into the part with the promoter and the sequence for easy
      # processing
      gene = [field.strip() for field in line.split(' ')]
      dna = DNA(gene[0],gene[1])
      # slow way of growing a list but it works for this purpose
      data_list.append(dna)



def main(parser):
  """ Drives the program.
  """
  args = parser.parse_args()
  train_data = []
  # read the file
  read_file(train_data, args.train)

  validation_data = []
  read_file(validation_data, args.validation)

  classify.classify(decision_tree,
                    validation_data,
                    False, str(args.confidence),
                    args.ipython)


if __name__ == "__main__":
  """Main entry point, only parses args and passes them on
  """
  parser = argparse.ArgumentParser(
    description =
    "Implements a naieve baysien classifier to classify text documents")

  parser.add_argument(
      "-t",
      "--train",
      help = 'the data on which you wish to train e.g. \"../data/training.txt\" ',
      required=True
      )
  parser.add_argument(
      '-v',
      '--validation',
      help = 'the validation data',
      required=True)
  parser.add_argument(
      '--ipython',
      help='this is an ipython session and we want to draw the figs, not save them',
      action='store_true')
  parser.add_argument(
      '-x',
      '--confidence',
      help='threshold confidence level for growing the decision tree. Can either be (0, 95, 99)',
      type=int
      )
  main(parser)
