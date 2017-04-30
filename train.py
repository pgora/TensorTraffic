# coding=utf-8
# Copyright 2017

# To use GPU, you need to compile Tensorflow following:
# http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html

# Tested on Tensorflow 0.12 and TensorFlow 1.0

#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division

import pprint
import itertools
import logging
import numpy as np
import pandas as pd
import os
import pdb
import shutil
import sys
import tensorflow as tf

from optparse import OptionParser
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.training import saver as saver_lib
import time

def xrange(x):
    return iter(range(x))

""" Modified ValdationMonitor from Tensorflow 0.12 """
""" To implement regression metric """
class RegressionMonitor(tf.contrib.learn.monitors.EveryN):
  """Runs evaluation of a given estimator, at most every N steps.
  Note that the evaluation is done based on the saved checkpoint, which will
  usually be older than the current step.
  Can do early stopping on validation metrics if `early_stopping_rounds` is
  provided.
  """

  def __init__(self, x=None, y=None, input_fn=None, batch_size=None,
               eval_steps=None,
               every_n_steps=1000, metrics=None, early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, name=None):
    """Initializes a ValidationMonitor.
    Args:
      x: See `BaseEstimator.evaluate`.
      y: See `BaseEstimator.evaluate`.
      input_fn: See `BaseEstimator.evaluate`.
      batch_size: See `BaseEstimator.evaluate`.
      eval_steps: See `BaseEstimator.evaluate`.
      every_n_steps: Check for new checkpoints to evaluate every N steps. If a
          new checkpoint is found, it is evaluated. See `EveryN`.   # question: does it take a trained model from checkpoint?
      metrics: See `BaseEstimator.evaluate`.
      early_stopping_rounds: `int`. If the metric indicated by
          `early_stopping_metric` does not change according to
          `early_stopping_metric_minimize` for this many steps, then training
          will be stopped.
      early_stopping_metric: `string`, name of the metric to check for early
          stopping.
      early_stopping_metric_minimize: `bool`, True if `early_stopping_metric` is
          expected to decrease (thus early stopping occurs when this metric
          stops decreasing), False if `early_stopping_metric` is expected to
          increase. Typically, `early_stopping_metric_minimize` is True for
          loss metrics like mean squared error, and False for performance
          metrics like accuracy.
      name: See `BaseEstimator.evaluate`.
    Raises:
      ValueError: If both x and input_fn are provided.
    """
    super(RegressionMonitor, self).__init__(every_n_steps=every_n_steps,
                                            first_n_steps=-1)
    # TODO(mdan): Checks like this are already done by evaluate.
    if x is None and input_fn is None:
      raise ValueError("Either x or input_fn should be provided.")
    self.x = x
    self.y = y
    self.input_fn = input_fn
    self.batch_size = batch_size
    self.eval_steps = eval_steps
    self.metrics = metrics
    self.early_stopping_rounds = early_stopping_rounds
    self.early_stopping_metric = early_stopping_metric
    self.early_stopping_metric_minimize = early_stopping_metric_minimize
    self.name = name
    self._best_value_step = None
    self._best_value = None
    self._early_stopped = False
    self._latest_path = None
    self._latest_path_step = None

    # Every 10k steps, we evaluate if the result is better or not.
    # If it's not much better, stop the execution.
    self._past_best_big_checkpoint = 100.0
    self._minimal_improvement_treshold = 0.01
    self._criteria_check_interval = 10000

  @property
  def early_stopped(self):
    """Returns True if this monitor caused an early stop."""
    return self._early_stopped

  @property
  def best_step(self):
    """Returns the step at which the best early stopping metric was found."""
    return self._best_value_step

  @property
  def best_value(self):
    """Returns the best early stopping metric value found so far."""
    return self._best_value

  def every_n_step_end(self, step, outputs):
    super(RegressionMonitor, self).every_n_step_end(step, outputs) # does it do anything now ?
    # TODO(mdan): The use of step below is probably misleading.
    # The code should probably use the step from the checkpoint, because
    # that's what is being evaluated.
    if self._estimator is None:
      raise ValueError("Missing call to set_estimator.")
    # Check that we are not running evaluation on the same checkpoint.
    latest_path = saver_lib.latest_checkpoint(self._estimator.model_dir)
    if latest_path is None:
      logging.debug("Skipping evaluation since model has not been saved yet "
                    "at step %d.", step)
      return False
    if latest_path is not None and latest_path == self._latest_path:
      logging.debug("Skipping evaluation due to same checkpoint %s for step %d "
                    "as for step %d.", latest_path, step,
                    self._latest_path_step)
      return False
    self._latest_path = latest_path
    self._latest_path_step = step
    # Run evaluation and log it.
    stats = evaluate(self._estimator, self.x, self.y)

    print ( "Validation (step %d): AVG_ERR: %s %%  MAX_ERR: %s %%" %  (step, \
       stats['relative_avg_err'] * 100, stats['relative_max_err'] * 100))

    if (step / 1000) % (self._criteria_check_interval / 1000) == 0:
        # Stopping after not receiving progress bigger than 0.01% after 10k steps.
        if stats['relative_avg_err'] * 100 > \
            self._past_best_big_checkpoint - self._minimal_improvement_treshold:
            print("The relative average error is not improving. Stopping after %d steps" % step)
            return True
        else:
            print("The relative average error improved from %s %% to %s %% after 10k steps" \
              % (self._past_best_big_checkpoint, stats['relative_avg_err']*100))
            self._past_best_big_checkpoint = stats['relative_avg_err'] * 100
    return False

class ContextFilter(logging.Filter):
    def filter(self, record):
        # Remove noise. Get it back if something will not work.
        if 'is deprecated' in record.msg or 'not supported' in record.msg:
            return False
        else:
            return True

# This is to print DNNRegressor progress
f = ContextFilter()
logging.getLogger('tensorflow').setLevel(logging.INFO)
logging.getLogger('tensorflow').addFilter(f)

# Ovewritted by gridsearch.
PARAMS = {
  'learning_rate': 0.01,
  'layers': [100, 100],
  'max_steps': 100,
  'test_ratio': 0.2,
  'cv_folds': 5,
  'dropout': 0.0,
  'training_set_size': 90000,
  'div_const': 100,  # TODO: experiment with other values
}


# TODO globals not that good.
#ID = 0
batch_step = 0
batch_step_val = 0
batch_size = 10240
training_set_size = 81920
val_set_size = 20480

errors_main = []

def evaluate(classifier, X_test, y_test):
  pred_iterable = classifier.predict(input_fn = lambda: predict_input_fn(X_test))
  #errors = []
  pred = []
  for i in range(y_test.size):
    #p = next(pred_iterable)
    #y = next(y_test)
    pred.append(next(pred_iterable))
    #pred.append(p)
    #errors.append(np.abs(p-y))
    #print np.abs(y-p)

  stats = get_stats(np.asarray(pred), y_test)
  #errors_main = errors
  return stats

def eval_test(classifier):
  all_data = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename="micro_data_final_test.csv",
    target_dtype=np.float32,
    features_dtype=np.float32)  # target_column=-1

  # TODO: When we have more signals then only traffic lights, we shouldn't
  # divide everything by 120
  normalized_data = all_data.data
  assert normalized_data.shape[1] in (15, 17)
  div_row = [120.0 for i in xrange(15)]
  if normalized_data.shape[1] == 17:  # 15 traffic signal settings + number of cars + number of new cars
    print("Running on MICRO data")
    div_row.extend([100000.0, 20.0])
  else:
    print("Running on MESO data")

  # normalized_data = (normalized_data - np.mean(normalized_data, axis=0, keepdims=True)) / np.std(normalized_data, axis=0, keepdims=True)
  normalized_data = normalized_data / div_row

  # TODO: experiment with other constants. In the old code we divide target by 100
  normalized_target = all_data.target / PARAMS['div_const']

  pred_iterable = classifier.predict(input_fn=lambda: predict_input_fn(all_data))
  pred = []
  for i in pred_iterable:
    pred.append(i)

  y = normalized_target
  errors = zip(y,
               pred,
               np.subtract(y, pred) * PARAMS['div_const'],
               np.subtract(y, pred) / y)
  df = pd.DataFrame(np.array(list(errors)),
                    columns=["y", "pred", "absolute_error", "relative_error"])
  df.to_csv("model1.csv", index=False)

  return pred


def aggregate_results(results):
  print("AGGREGATE RESULTS: ", results)
  ret = {}
  for key in results[0].keys():
    s = 0
    for res in results:
      s += res[key]
    ret[key] = s / len(results)
  return ret

def input_fn(x_train, y_train):
  global batch_step
  feature_cols = tf.constant(x_train)
  labels = tf.constant(y_train)

  # TODO every 10th batch will be smaller ¯\_(ツ)_/¯
  i = batch_step % training_set_size
  j = min(i+batch_size, training_set_size)
  batch_step += batch_size
  return feature_cols[i:j], labels[i:j]

def predict_input_fn(x_train):
    feature_cols = tf.constant(x_train)
    return feature_cols

# not used now
#def valid_fn(x_test, y_test):
#  feature_cols = tf.constant(x_test)
#  labels = tf.constant(y_test)
#  return feature_cols, labels

def get_stats(pred, expected):
  assert pred.shape == expected.shape
  ret = {}
  # All target values were divided by div_const to avoid overflow and
  # improve training. For computing absolute error, we need to multiply
  # by this const.
  const = PARAMS['div_const']
  # Absolute errors
  ret['absolute_max_err'] = np.max(np.abs(pred-expected) * const)
  ret['absolute_avg_err'] = np.mean(np.abs(pred-expected) * const)
  ret['absolute_avg_err'] = np.mean(np.abs(pred-expected) * const)
  # Relative errors
  ret['relative_max_err'] = np.max(np.abs(pred-expected) / expected)
  ret['relative_avg_err'] = np.mean(np.abs(pred-expected) / expected)

  return ret

def do_training(id, X_train, X_test, y_train, y_test):
  global batch_step
  global training_set_size
  #global ID

  MODEL_DIR = "model_" + str(id)

  if os.path.isdir(MODEL_DIR):
    print("Removing old model dir...")
    shutil.rmtree(MODEL_DIR)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column(
                      "", dimension=X_train.shape[1])]
  batch_step = 0
  training_set_size = PARAMS['training_set_size']
  config = run_config.RunConfig(log_device_placement=False, \
   save_checkpoints_secs=5)
  classifier = tf.contrib.learn.DNNRegressor( # activation_fn: tf.nn.relu by default
      feature_columns=feature_columns,
      hidden_units=PARAMS['layers'],
      model_dir=MODEL_DIR,
      optimizer=tf.train.AdamOptimizer(learning_rate=PARAMS['learning_rate'], epsilon=0.8),
      dropout=PARAMS['dropout'],
      config=config)

  monitor = RegressionMonitor(x=X_test, y=y_test)
  classifier.fit(input_fn = lambda: input_fn(X_train,y_train),
                 steps=PARAMS['max_steps'], monitors=[monitor])

  # Evaluate accuracy.
  errors = eval_test(classifier)
  for err in errors:
    print(err)

  return evaluate(classifier, X_test, y_test)

def cross_valid(normalized_X, normalized_y):
  print("Running cross-validation with params: ", PARAMS)
  kf = KFold(n_splits=PARAMS['cv_folds'], shuffle=True, random_state=42)

  results = []
  for it, (train_index, test_index) in enumerate(kf.split(normalized_X)):
    print("Iteration %d..." % it)
    X_train, X_test = normalized_X[train_index], normalized_X[test_index]
    y_train, y_test = normalized_y[train_index], normalized_y[test_index]

    res = do_training(it, X_train, X_test, y_train, y_test)
    results.append(res)

  ret = aggregate_results(results)
  print("Aggregated CV results: ")
  pprint.pprint(ret)
  return ret

def single_run(normalized_X, normalized_y):
  X_train, X_test, y_train, y_test = train_test_split(
      normalized_X, normalized_y, test_size=PARAMS['test_ratio'], random_state=42)
  print(do_training(1, X_train, X_test, y_train, y_test))

def create_result_file_name(id, params):
  #global ID
  return ('result_' + str(id) + '_' + str(params['layers']) + '_' + str(params['training_set_size']) + \
  '_' + str(params['learning_rate']) + '_' + str(params['dropout'])).replace(" ", "_")

def check_grid(normalized_X, normalized_y):
  #global ID
  X_train, X_test, y_train, y_test = train_test_split(
      normalized_X, normalized_y, test_size=PARAMS['test_ratio'], random_state=42)

  # ----------------------------------------------------------------------------
  # Grid search configurations
  # ----------------------------------------------------------------------------

#  lr = [0.0001, 0.001, 0.01, 0.1]
#  layers = [[100, 100, 100], [100, 200, 100], [200, 300, 200], [300, 400, 300], \
#    [100,150,200,150,100], [50, 100, 200, 300, 200, 100, 50]]
#  dropout = [0.05, 0.1, 0.15, 0.2]
#  max_steps = [1000000]
#  training_set_size_list = [81920, 30720, 10240]

  lr = [0.01]
  layers = [[100, 200, 100]]
  dropout = [0.05]
  max_steps = [1000000]
  training_set_size_list = [10240]



  # ----------------------------------------------------------------------------
  # Grid search configurations
  # ----------------------------------------------------------------------------

  results = {}
  for i, x in enumerate(itertools.product(lr, layers, dropout, max_steps, \
    training_set_size_list)):

    #ID = i
    #lock_filename = 'lock-' + str(i) + '.lck'
    #if os.path.exists(lock_filename):
    #    continue
    #open(lock_filename, 'a').close()

    # Epochs trained: 15k. Batch size: 1k Training set size: 1k/2k/5k/10k.
    PARAMS['max_steps'] = x[3]
    PARAMS['learning_rate'] = x[0]
    PARAMS['layers'] = x[1]
    PARAMS['dropout'] = x[2]
    PARAMS['training_set_size'] = x[4]
    print("==========Running params:", PARAMS)
    sys.stdout.flush()

    start = time.time()
    ret = do_training(i, X_train, X_test, y_train, y_test)
    end = time.time()
    ret['training_time_100_steps'] = end-start
    print("==========Results for params:", PARAMS)

    with open(create_result_file_name(i, PARAMS) + '.csv', 'w') as the_file:
      the_file.write(str(PARAMS['layers']) + ',' + str(PARAMS['training_set_size']) + \
      ',' + str(PARAMS['learning_rate']) + ',' + str(PARAMS['dropout']) + ',')
      the_file.write(str(ret['relative_max_err']) + ',' + str(ret['relative_avg_err']))

    print (ret)
    sys.stdout.flush()
    results[str(x)] = ret

  print("FINAL RESULTS:")
  pprint.pprint(results)

def main():
  usage = "Usage: %prog [options] <path_to_train_data>"
  parser = OptionParser(usage)
  parser.add_option("-c", "--cross_valid",
                    action="store_true", help="Run cross-validation with 5 folds.")
  parser.add_option("-g", "--grid",
                    action="store_true", help="Run grid search on params.")

  options, args = parser.parse_args()
  if len(args) != 1:
    print("Usage: %s [options] <path_to_train_data>" % sys.argv[0])
    sys.exit(1)

  data_path = args[0]
  all_data = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=data_path,
      target_dtype=np.float32,
      features_dtype=np.float32)  # target_column=-1

  # TODO: When we have more signals then only traffic lights, we shouldn't
  # divide everything by 120
  normalized_data = all_data.data
  assert normalized_data.shape[1] in (15, 17)
  div_row = [120.0 for i in xrange(15)]
  if normalized_data.shape[1] == 17:  # 15 traffic signal settings + number of cars + number of new cars
    print("Running on MICRO data")
    div_row.extend([100000.0, 20.0])
  else:
    print("Running on MESO data")

  #normalized_data = (normalized_data - np.mean(normalized_data, axis=0, keepdims=True)) / np.std(normalized_data, axis=0, keepdims=True)
  normalized_data = normalized_data / div_row

  # TODO: experiment with other constants. In the old code we divide target by 100
  normalized_target = all_data.target / PARAMS['div_const']

  if options.grid:
    check_grid(normalized_data, normalized_target)
  else:
    if options.cross_valid:
      print("Running Cross-Validation...")
      cross_valid(normalized_data, normalized_target)
    else:
      print("Running single training...")
      single_run(normalized_data, normalized_target)


if __name__ == '__main__':
  main()
