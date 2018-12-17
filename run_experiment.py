from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.apputils import app
from tensorflow import gfile
import gflags as flags
from time import gmtime
from time import strftime

import os
import pickle
import numpy as np
import sys

from utils import utils
from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler

flags.DEFINE_string("dataset", "mnist", "Dataset name")
flags.DEFINE_string("sampling_method", "margin",
                    ("Name of sampling method to use, can be any defined in "
                     "AL_MAPPING in sampling_methods.constants"))
flags.DEFINE_string(
    "select_method", "None",
    "Method to use for selecting points.")
flags.DEFINE_string(
    "score_method", "logistic",
    "Method to use to calculate accuracy.")

# ================= static params =================================
flags.DEFINE_float(
    "warmstart_size", 128,
    ("Can be float or integer.  Float indicates percentage of training data "
     "to use in the initial warmstart model"))
flags.DEFINE_float(
    "batch_size", 32,
    ("Can be float or integer.  Float indicates batch size as a percentage "
     "of training data size."))
flags.DEFINE_integer("trials", 2,
                     "Number of curves to create using different seeds")
flags.DEFINE_float("max_dataset_size", 0.4,
                    ("maximum number of datapoints to include in data "
                     "zero indicates no limit"))
flags.DEFINE_integer("seed", 1, "Seed to use for rng and random state")
flags.DEFINE_string("do_save", "True",
                    "whether to save log and results")
flags.DEFINE_string("save_dir", "./toy_experiments",
                    "Where to save outputs")
flags.DEFINE_string("data_dir", "/tmp/data",
                    "Directory with predownloaded and saved datasets.")

FLAGS = flags.FLAGS

# ======================================================

def generate_one_curve(X,
                       y,
                       sampler,
                       score_model,
                       seed,
                       warmstart_size,
                       batch_size,
                       select_model=None,
                       max_points=None):
  """Creates one learning curve for both active and passive learning.

  Will calculate accuracy on validation set as the number of training data
  points increases for both PL and AL.
  Caveats: training method used is sensitive to sorting of the data so we
    resort all intermediate datasets

  Args:
    X: training data
    y: training labels
    sampler: sampling class from sampling_methods, assumes reference
      passed in and sampler not yet instantiated.
    score_model: model used to score the samplers.  Expects fit and predict
      methods to be implemented.
    seed: seed used for data shuffle and other sources of randomness in sampler
      or model training
    warmstart_size: float or int.  float indicates percentage of train data
      to use for initial model
    batch_size: float or int.  float indicates batch size as a percent of
      training data
    select_model: defaults to None, in which case the score model will be
      used to select new datapoints to label.  Model must implement fit, predict
      and depending on AL method may also need decision_function.
    confusion: percentage of labels of one class to flip to the other
    max_points: limit dataset size for preliminary
    standardize_data: wheter to standardize the data to 0 mean unit variance
    norm_data: whether to normalize the data.  Default is False for logistic
      regression.
    train_horizon: how long to draw the curve for.  Percent of training data.

  Returns:
    results: dictionary of results for all samplers
    sampler_states: dictionary of sampler objects for debugging
  """

  def select_batch(sampler, N, already_selected, **kwargs):
    kwargs["N"] = N
    kwargs["already_selected"] = already_selected
    batch_AL = sampler.select_batch(**kwargs)
    return batch_AL

  np.random.seed(seed)
  data_splits = [2./3, 1./6, 1./6]

  # 2/3 of data for training
  if max_points is None:
    max_points = len(y)
  if max_points < 1:
    max_points = int(max_points * len(y))
  else:
    max_points = int(max_points)
  train_size = int(min(max_points, len(y) * data_splits[0]))
  if batch_size < 1:
    batch_size = int(batch_size * train_size)
  else:
    batch_size = int(batch_size)
  if warmstart_size < 1:
    seed_batch = int(warmstart_size * train_size)
  else:
    seed_batch = int(warmstart_size)
  seed_batch = max(seed_batch, 6 * len(np.unique(y)))

  indices, X_train, y_train, X_val, y_val, X_test, y_test = (
      utils.get_train_val_test_splits(X, y, max_points,seed,
                                      seed_batch, split=data_splits))

  print(" warmstart batch: " +
        str(seed_batch) + " batch size: " + str(batch_size) + " seed: " + str(seed))

  # Initialize samplers
  sampler = sampler(X_train, seed)

  results = {}
  data_sizes = []
  accuracy = []
  selected_inds = range(seed_batch)

  # If select model is None, use score_model
  same_score_select = False
  if select_model is None:
    select_model = score_model
    same_score_select = True

  n_batches = int(np.ceil((train_size - seed_batch) *
                          1.0 / batch_size)) + 1
  for b in range(n_batches):
    n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
    print("Training model on " + str(n_train) + " datapoints")

    assert n_train == len(selected_inds)
    data_sizes.append(n_train)

    # Sort active_ind so that the end results matches that of uniform sampling
    partial_X = X_train[sorted(selected_inds)]
    partial_y = y_train[sorted(selected_inds)]
    score_model.fit(partial_X, partial_y)
    if not same_score_select:
      select_model.fit(partial_X, partial_y)
    acc = score_model.score(X_test, y_test)
    accuracy.append(acc)
    print("Sampler: %s, Accuracy: %.2f%%" % (sampler.name, accuracy[-1]*100))

    n_sample = min(batch_size, train_size - len(selected_inds))
    select_batch_inputs = {
        "model": select_model,
        "labeled": dict(zip(selected_inds, y_train[selected_inds])),
        "eval_acc": accuracy[-1],
        "X_test": X_val,
        "y_test": y_val,
        "y": y_train
    }
    new_batch = select_batch(sampler, n_sample, selected_inds, **select_batch_inputs)
    selected_inds.extend(new_batch)
    print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
    assert len(new_batch) == n_sample
    assert len(list(set(selected_inds))) == len(selected_inds)

  # Check that the returned indice are correct and will allow mapping to
  # training set from original data

  results["accuracy"] = accuracy
  results["selected_inds"] = selected_inds
  results["data_sizes"] = data_sizes
  results["indices"] = indices

  return results, sampler


def main(argv):
    del argv

    if not gfile.Exists(FLAGS.save_dir):
        try:
            gfile.MkDir(FLAGS.save_dir)
        except:
            print(('WARNING: error creating save directory, '))

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.dataset + '_' + FLAGS.sampling_method)

    if FLAGS.do_save == "True":
        if not gfile.Exists(save_dir):
            try:
                gfile.MkDir(save_dir)
            except:
                print(('WARNING: error creating save directory, '
                       'directory most likely already created.'))

        # Set up logging
        filename = os.path.join(
            save_dir, "log-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".txt")
        sys.stdout = utils.Logger(filename)

    X, y = utils.get_mldata(FLAGS.data_dir, FLAGS.dataset) #load dataset!
    starting_seed = FLAGS.seed

    all_results = {}

    for seed in range(starting_seed , starting_seed + FLAGS.trials):
        sampler = get_AL_sampler(FLAGS.sampling_method)             #load sampler!
        score_model = utils.get_model(FLAGS.score_method, seed)     #load score model!
        if (FLAGS.select_method == "None" or                        #load select model!
            FLAGS.select_method == FLAGS.score_method):
            select_model = None
        else:
            select_model = utils.get_model(FLAGS.select_method, seed)

        results, sampler_state = \
        generate_one_curve(X=X,
                           y=y,
                           sampler=sampler,
                           score_model=score_model,
                           seed=seed,
                           warmstart_size=FLAGS.warmstart_size,
                           batch_size=FLAGS.batch_size,
                           select_model=select_model,
                           max_points=FLAGS.max_dataset_size)

        key = (FLAGS.dataset, FLAGS.sampling_method, FLAGS.score_method,
               FLAGS.select_method, FLAGS.warmstart_size, FLAGS.batch_size, seed)

        #sampler_output = sampler_state.to_dict()
        #results['sampler_output'] = sampler_output
        results['sampler_output'] = None
        all_results[key] = results

    fields = ['dataset', 'sampling_methods', 'score_method', 'select_method',
              'warmstart size', 'batch size', 'seed']
    all_results['tuple_keys'] = fields

    if FLAGS.do_save == "True":
        filename = ("results_score_" + FLAGS.score_method +
                    "_select_" + FLAGS.select_method)
        existing_files = gfile.Glob(os.path.join(save_dir, filename + "*.pkl"))
        filename = os.path.join(save_dir,
                                filename + "_" + str(1000 + len(existing_files))[1:] + ".pkl")
        pickle.dump(all_results, gfile.GFile(filename, "w"))
        sys.stdout.flush_file()


if __name__ == "__main__":
  app.run()
