import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import kerastuner as kt
import random
import csv

ratio_split = [0.75,0.9,1.0]

seed = 111
random.seed(seed)
rng = np.random.default_rng(seed)
rng.random(seed)
tf.random.set_seed(seed)

print(tf.__version__)

def create_report(arr, filename,title):
    df = pd.DataFrame(arr)
    print(df)
    plt.clf()
    plt.figure(0)
    plt.plot(df['size'], df['train_f1'], color ='steelblue')
    plt.plot(df['size'], df['val_f1'], color ='red')
    plt.plot(df['size'], df['train_f1'],'o', color ='steelblue')
    plt.plot(df['size'], df['val_f1'],'o', color = 'red')
    plt.xlabel(" Fraction of data")
    plt.ylabel("f1_score")
    plt.legend(['Train f1', 'Validation f1'])
    plt.ylim(0.3, 1.0)
    plt.title(title)
    plt.savefig(filename)
    return True

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0.0, 1.0)))
        Positives = K.sum(K.round(K.clip(y_true, 0.0, 1.0)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0.0, 1.0)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0.0, 1.0)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall    = recall(y_true, y_pred)

    return 2.0 * ((precision * recall) / (precision + recall + K.epsilon()))

def data_size_tests(tuner, best_hp, train_df, val_df):
    performance_arr = []
    num_steps = 10
    for i in range(1, num_steps + 1):
        my_dict = {}
        my_dict['size'] = i / num_steps
        df_train_part = train_df[0:(round)(len(train_df)* i /num_steps)]
        df_val_part   = val_df[0:(round)(len(train_df)* i /num_steps)]
        train_ds_part = tfdf.keras.pd_dataframe_to_tf_dataset(df_train_part, label=label)
        val_ds_part   = tfdf.keras.pd_dataframe_to_tf_dataset(df_val_part,   label=label)
        model = tuner.hypermodel.build(best_hp)
        model.compile(metrics=[f1_score])
        model.fit(x = train_ds_part)
        my_dict['train_f1']  = model.evaluate(x = train_ds_part)[1]
        my_dict['val_f1']    = model.evaluate(x = val_ds_part)[1]
        performance_arr.append(my_dict)
    return performance_arr

def split_dataset(dataset, ratio=ratio_split):
  indx = np.random.rand(len(dataset))
  train_indexes = (indx <= ratio[0])
  val_indexes   = (indx <= ratio[1]) & (indx>ratio[0])
  test_indexes  = (indx  > ratio[1])
  return dataset[train_indexes], dataset[val_indexes], dataset[test_indexes]


def create_rf_model(hp):
    model = tfdf.keras.RandomForestModel(
        num_trees=hp.Int('num_trees', min_value=10, max_value=300, step=10),
        max_depth=hp.Int('max_depth', min_value=2, max_value=20, step=1),
        random_seed=seed)
    model.compile(metrics = [f1_score])
    return model

def create_bt_model(hp):
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=hp.Int('num_trees', min_value=10, max_value=500, step=20),
        growing_strategy=hp.Choice('growing_strategy', values=['BEST_FIRST_GLOBAL', 'LOCAL']),
        subsample=hp.Float('subsample', min_value=0.1, max_value=0.95, step=0.05),
        max_depth=hp.Int('max_depth', min_value=2, max_value=20, step=1),
        random_seed = seed)
    model.compile(metrics = [f1_score])
    return model

def evaluate(tuner_rf, best_hp, path):
    model = tuner_rf.hypermodel.build(best_hp)
    model.compile(metrics=[f1_score])
    model.fit(train_ds)
    f1_score_test = model.evaluate(x=test_ds)[1]
    f1_score_train = model.evaluate(x=train_ds)[1]
    f1_score_val = model.evaluate(x=val_ds)[1]

    print("F1 score for the Random Forest model ,train,val,test : ", f1_score_train, f1_score_val, f1_score_test)
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["train", "val", "test", f1_score_train, f1_score_val, f1_score_test])
    return True

path_filename = "../data/data.csv"
df  = pd.read_csv(path_filename)
df = df.drop(columns=['customerID'])
print(df.head())

#
# Cleaning the data:
#

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df.dropna(inplace = True)

label = "Churn"
classes = df[label].unique().tolist()
print(f"Label classes: {classes}")
print("Label class elements ", label, "  ", len(df[df[label]=="No"]))
print("Label class elements", label, "  ",  len(df[df[label]=="Yes"]))
df[label] = df[label].map(classes.index)

df = pd.get_dummies(df)
for col in df.columns:
    print(col)
df = df.astype(float)
print(df.head())

train_ds_pd, val_ds_pd,  test_ds_pd = split_dataset(df)

print("Length training data",len(train_ds_pd))
print("Length validation dataset", len(val_ds_pd))
print("Length test dataset", len(test_ds_pd))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
val_ds   = tfdf.keras.pd_dataframe_to_tf_dataset(val_ds_pd, label=label)
test_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)
################################
#
# Random forest algorithm
#
################################

project_name = 'random_forest'

tuner_rf = kt.RandomSearch(
    create_rf_model,
    objective    = kt.Objective("val_f1_score", direction="max"),
    max_trials   = 100,
    project_name= project_name
)


tuner_rf.search(x                 = train_ds,
                validation_data   = val_ds,
                epochs            = 1,
                random_seed       = seed
                )

best_hp = tuner_rf.get_best_hyperparameters()[0]
perf_arr = data_size_tests(tuner_rf, best_hp, train_ds_pd, val_ds_pd)
print("\n\n\n\nBest hp :",best_hp)
create_report(perf_arr, "reports/random_forest.png", "Random Forest Algorithm")
evaluate(tuner_rf, best_hp,os.path.join(project_name,'test_results.csv') )


#
# GradientBoostedTrees model
#

project_name = 'gradient_boosted_trees'
tuner_bt = kt.RandomSearch(
    create_bt_model,
    objective  = kt.Objective("val_f1_score" , direction="max"),
    max_trials = 100,
    project_name =  project_name)

tuner_bt.search(x         = train_ds,
                epochs    = 1,
                validation_data = val_ds,
                metrics         = ["val_f1_score"],
                verbose         = 0,
                project_name    = project_name
                )

best_hp = tuner_bt.get_best_hyperparameters()[0]
perf_arr = data_size_tests(tuner_bt, best_hp,train_ds_pd, val_ds_pd)
print("\n\n\n\nBest hp :",best_hp)

create_report(perf_arr, "reports/gradien_boosting.png", "Gradient Boosted Trees Algorithm")
evaluate(tuner_bt, best_hp,os.path.join(project_name,'test_results.csv') )
