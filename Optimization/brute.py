import collections
import tensorflow as tf
import numpy as np
from random import randint
import time
from tensorflow.contrib.learn.python.learn.estimators import run_config
tf.logging.set_verbosity(tf.logging.ERROR)

def normalize(data):
    x_train = list(data)
    x_train.append(30000)
    x_train.append(20)
    div_row = [120.0 for i in xrange(15)]
    div_row.extend([100000.0, 20.0])
    normalized_data = np.asarray(x_train) / div_row
    return normalized_data.tolist()

def predict_input_fn(data):
    input = []
    for i in data:
        input.append(normalize(i))

    feature_tensor = tf.constant(input)
    return feature_tensor

MODEL_DIR="/tmp/road_model2"
config = run_config.RunConfig(log_device_placement=False, \
    save_checkpoints_secs=5)
classifier = tf.contrib.learn.DNNRegressor(
    feature_columns=[tf.contrib.layers.real_valued_column("", dimension=15)],
    hidden_units=[100, 200, 100],
    model_dir=MODEL_DIR,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.8),
    dropout=0.05,
    config=config)

best_conf = []
best_score = 100000

def eval_batch_dnn():
    global best_conf
    global best_score

    data = []
    for i in range(10240):
        genome = []
        for j in range(15):
            genome.append(randint(0, 119))
        data.append(genome)
    pred_iterable = classifier.predict(input_fn = lambda: predict_input_fn(data))
    for i in range(10240):
        val = next(pred_iterable) * 100
        if (val < best_score):
            best_score = val
            best_conf = data[i]
            print "New best score = " + str(best_score) + " for " + str(best_conf)

print "Starting random search."
start = time.time()
for i in range(1000):
    eval_batch_dnn()
    if i > 0 and i % 10 == 0:
        print "Processed " + str(1.0 * i / 100) + "m random combinations in " \
        + str(time.time() - start) + " seconds"
