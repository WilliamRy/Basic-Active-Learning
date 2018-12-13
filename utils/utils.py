from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from tensorflow import gfile
import numpy as np
import copy
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class Logger(object):
  """Logging object to write to file and stdout."""

  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = gfile.GFile(filename, "w")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()

  def flush_file(self):
    self.log.flush()

def get_mldata(data_dir, name):
    """Loads data from data_dir.

      Looks for the file in data_dir.
      Assumes that data is in pickle format with dictionary fields data and target.

      Args:
        data_dir: directory to look in
        name: dataset name, assumes data is saved in the save_dir with filename
          <name>.pkl
      Returns:
        data and targets
      Raises:
        NameError: dataset not found in data folder.
      """
    dataname = name
    filename = os.path.join(data_dir, dataname + ".pkl")
    if not gfile.Exists(filename):
        raise NameError("ERROR: dataset not available")
    data = pickle.load(gfile.GFile(filename, "r"))
    X = data["data"]
    y = data["target"]

    return X, y

def get_model(method, seed=13):
  """Construct sklearn model using either logistic regression or linear svm.

  Wraps grid search on regularization parameter over either logistic regression
  or svm, returns constructed model

  Args:
    method: string indicating scikit method to use, currently accepts logistic
      and linear svm.
    seed: int or rng to use for random state fed to scikit method

  Returns:
    scikit learn model
  """
  if method == "logistic":
    model = LogisticRegression(random_state=seed, multi_class="multinomial",
                               solver="lbfgs", max_iter=200)
    params = {"C": [10.0**(i) for i in range(-3, 1)]}

  else:
    raise NotImplementedError("ERROR: " + method + " not implemented")

  model = GridSearchCV(model, params, cv=3)
  return model

def get_train_val_test_splits(X, y, max_points, seed, seed_batch,
                              split=(2./3, 1./6, 1./6)):
  """Return training, validation, and test splits for X and y.

  Args:
    X: features
    y: targets
    max_points: # of points to use when creating splits.
    seed: seed for shuffling.
    confusion: labeling noise to introduce.  0.1 means randomize 10% of labels.
    seed_batch: # of initial datapoints to ensure sufficient class membership.
    split: percent splits for train, val, and test.
  Returns:
    indices: shuffled indices to recreate splits given original input data X.
    y_noise: y with noise injected, needed to reproduce results outside of
      run_experiments using original data.
  """
  confusion = 0
  np.random.seed(seed)
  X_copy = copy.copy(X)
  y_copy = copy.copy(y)

  # Introduce labeling noise
  y_noise = flip_label(y_copy, confusion)

  indices = np.arange(len(y))

  if max_points is None:
    max_points = len(y_noise)
  else:
    max_points = min(len(y_noise), max_points)
  train_split = int(max_points * split[0])
  val_split = train_split + int(max_points * split[1])
  assert seed_batch <= train_split

  # Do this to make sure that the initial batch has examples from all classes
  min_shuffle = 3
  n_shuffle = 0
  y_tmp = y_noise

  # Need at least 4 obs of each class for 2 fold CV to work in grid search step
  while (any(get_class_counts(y_tmp, y_tmp[0:seed_batch]) < 4)
         or n_shuffle < min_shuffle):
    np.random.shuffle(indices)
    y_tmp = y_noise[indices]
    n_shuffle += 1

  X_train = X_copy[indices[0:train_split]]
  X_val = X_copy[indices[train_split:val_split]]
  X_test = X_copy[indices[val_split:max_points]]
  y_train = y_noise[indices[0:train_split]]
  y_val = y_noise[indices[train_split:val_split]]
  y_test = y_noise[indices[val_split:max_points]]
  # Make sure that we have enough observations of each class for 2-fold cv
  assert all(get_class_counts(y_noise, y_train[0:seed_batch]) >= 4)
  # Make sure that returned shuffled indices are correct
  assert all(y_noise[indices[0:max_points]] ==
             np.concatenate((y_train, y_val, y_test), axis=0))
  return (indices[0:max_points], X_train, y_train,
          X_val, y_val, X_test, y_test)

def get_class_counts(y_full, y):
  """Gets the count of all classes in a sample.

  Args:
    y_full: full target vector containing all classes
    y: sample vector for which to perform the count
  Returns:
    count of classes for the sample vector y, the class order for count will
    be the same as long as same y_full is fed in
  """
  classes = np.unique(y_full)
  classes = np.sort(classes)
  unique, counts = np.unique(y, return_counts=True)
  complete_counts = []
  for c in classes:
    if c not in unique:
      complete_counts.append(0)
    else:
      index = np.where(unique == c)[0][0]
      complete_counts.append(counts[index])
  return np.array(complete_counts)

def flip_label(y, percent_random):
  """Flips a percentage of labels for one class to the other.

  Randomly sample a percent of points and randomly label the sampled points as
  one of the other classes.
  Does not introduce bias.

  Args:
    y: labels of all datapoints
    percent_random: percent of datapoints to corrupt the labels

  Returns:
    new labels with noisy labels for indicated percent of data
  """
  classes = np.unique(y)
  y_orig = copy.copy(y)
  indices = range(y_orig.shape[0])
  np.random.shuffle(indices)
  sample = indices[0:int(len(indices) * 1.0 * percent_random)]
  fake_labels = []
  for s in sample:
    label = y[s]
    class_ind = np.where(classes == label)[0][0]
    other_classes = np.delete(classes, class_ind)
    np.random.shuffle(other_classes)
    fake_label = other_classes[0]
    assert fake_label != label
    fake_labels.append(fake_label)
  y[sample] = np.array(fake_labels)
  assert all(y[indices[len(sample):]] == y_orig[indices[len(sample):]])
  return y