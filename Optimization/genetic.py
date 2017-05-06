from pyevolve import G1DList, GSimpleGA, Selectors
from pyevolve import Statistics, DBAdapters, Crossovers
from pyevolve import Initializators, Mutators
import collections
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators import run_config
tf.logging.set_verbosity(tf.logging.ERROR)

def predict_input_fn(x_train_1):
    x_train = list(x_train_1)
    x_train.append(30000)
    x_train.append(20)
    div_row = [120.0 for i in xrange(15)]
    div_row.extend([100000.0, 20.0])
    normalized_data = np.asarray(x_train) / div_row
    x_train = normalized_data.tolist()
    feature_tensor = tf.constant([x_train])
    return feature_tensor

MODEL_DIR="/tmp/road_model2"
config = run_config.RunConfig(log_device_placement=False, \
    save_checkpoints_secs=5)
classifier = tf.contrib.learn.DNNRegressor(
    feature_columns=[tf.contrib.layers.real_valued_column("", dimension=15)],
    hidden_units=[100, 200, 100],
    model_dir=MODEL_DIR,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01, epsilon=0.8),
    dropout=0.05,
    config=config)

def eval_func_dnn(genome):
    pred_iterable = classifier.predict(input_fn = lambda: predict_input_fn(genome))
    val = 100000 - next(pred_iterable) * 100
    return val

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes
def eval_func(genome):
    assert len(genome) == 15
    return eval_func_dnn(genome)

def run_main():

   print "Starting the experiment."
   print "Raw score is calculated as 100000 - predicted_score"
   print "Because the pyevolve can only maximize the fitness function"
   genome = G1DList.G1DList(15)
   genome.setParams(rangemin=0, rangemax=119)
   genome.mutator.set(Mutators.G1DListMutatorSwap)
   genome.mutator.add(Mutators.G1DListMutatorIntegerRange)
   genome.crossover.set(Crossovers.G1DListCrossoverUniform)
   genome.evaluator.set(eval_func)

   ga = GSimpleGA.GSimpleGA(genome)
   ga.setPopulationSize(900)
   ga.selector.set(Selectors.GTournamentSelector)
   ga.setGenerations(100)
   ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)

   sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
   ga.setDBAdapter(sqlite_adapter)
   ga.evolve(freq_stats=1)

   print "Best ga: "
   print ga.bestIndividual()

if __name__ == "__main__":
   run_main()
