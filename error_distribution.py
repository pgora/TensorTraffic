
# coding: utf-8

# In[17]:

from train import *
import pandas as pd
import numpy as np


# In[63]:

params = PARAMS
params['filename'] = "model1.csv"
params['max_steps'] = 1000000
params['learning_rate'] = 0.01
params['layers'] = [100, 200, 100]
params['dropout'] = 0.05
params['training_set_size'] = 90000


# In[64]:

all_data = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename="micro_data_train_valid.csv",
    target_dtype=np.float32,
    features_dtype=np.float32) 

X = all_data.data[:,:15]
y = all_data.target / PARAMS['div_const']
X = (X - np.mean(X, axis=0, keepdims=True))/np.std(X, axis=0, keepdims=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=PARAMS['test_ratio'], 
                                                    random_state=42)


# In[65]:

def pred(classifier, X_test, y_test):
    pred_iterable = classifier.predict(input_fn = lambda: predict_input_fn(X_test))
    pred = []
    for i in range(y_test.size):
        pred.append(next(pred_iterable))
        
    return pred
        
    

def train(X_train, y_train, X_test, y_test, params):
    global batch_step
    global training_set_size
    
    PARAMS['max_steps'] = params['max_steps']
    PARAMS['learning_rate'] = params['learning_rate']
    PARAMS['layers'] = params['layers']
    PARAMS['dropout'] = params['dropout']
    PARAMS['training_set_size'] = params['training_set_size']

    MODEL_DIR = "model_" + str(id)

    if os.path.isdir(MODEL_DIR):
        print("Removing old model dir...")
        shutil.rmtree(MODEL_DIR)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column(
                      "", dimension=X_train.shape[1])]
    batch_step = 0
    training_set_size = PARAMS['training_set_size']
    config = run_config.RunConfig(log_device_placement=False,                 save_checkpoints_secs=5)
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
    
    return y_test, pred(classifier, X_test, y_test)


# In[66]:

y, pred = train(X_train, y_train, X_test, y_test, params)


# In[71]:

errors = zip(y, 
             pred, 
             np.subtract(y, pred) * PARAMS['div_const'],
             np.subtract(y, pred) / y)
df = pd.DataFrame(np.array(list(errors)),
                 columns = ["y", "pred", "absolute_error", "relative_error"])


# In[72]:

df.to_csv(params["filename"], index=False)

