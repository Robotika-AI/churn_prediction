import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
import os
from tensorflow.keras import backend as K
import keras_tuner as kt
import warnings
import random
import tensorflow as tf
from copy import copy


warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ratio_split = [0.75,0.9,1.0]
seed = 111

def custom_f1(y_true, y_pred):
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

def create_report(arr, filename,title):
    df = pd.DataFrame(arr)
    print(df)
    plt.figure(0)
    plt.plot(df['size'], df['train_f1'], color ='steelblue')
    plt.plot(df['size'], df['val_f1'], color ='red')
    plt.plot(df['size'], df['train_f1'],'o', color ='steelblue')
    plt.plot(df['size'], df['val_f1'],'o', color = 'red')
    plt.xlabel(" Fraction of data")
    plt.ylabel("f1_score")
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.ylim(0.3, 1.0)
    plt.title(title)
    plt.savefig(filename)
    return True

"""
def data_size_tests(best_model,df):
    performance_arr = []
    num_steps = 10
    for i in range(1, num_steps + 1):
        print(i)
        my_dict = {}
        size_df = round(len(df) * i / num_steps)
        my_dict['size'] = i / num_steps
        df_part = df.sample(n=size_df, random_state=seed)
        train_ds_pd_part, val_ds_pd_part, test_ds_pd_part = split_dataset(df_part)
        train_ds_pd_part = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd_part, label=label)
        val_ds_pd_part = tfdf.keras.pd_dataframe_to_tf_dataset(val_ds_pd_part, label=label)
        best_model.fit(x = train_ds_pd_part)
        evaluation_train = best_model.evaluate(train_ds_pd_part, return_dict=True)
        my_dict['train_acc'] = evaluation_train['accuracy']

        evaluation_val = best_model.evaluate(val_ds_pd_part, return_dict=True)
        my_dict['val_acc'] = evaluation_val['accuracy']
        performance_arr.append(my_dict)
    return performance_arr

"""
def data_size_tests(tuner, best_hp, df):
    performance_arr = []
    num_steps = 10
    for i in range(1, num_steps + 1):
        my_dict = {}
        size_df = round(len(df) * i / num_steps)
        my_dict['size'] = i / num_steps
        df_part = df.sample(n=size_df, random_state=seed)
        train_ds_pd_part, val_ds_pd_part, test_ds_pd_part = split_dataset(df_part)
        print(train_ds_pd_part.head())
        print(i)
        train_ds_part = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd_part, label=label)
        val_ds_part   = tfdf.keras.pd_dataframe_to_tf_dataset(val_ds_pd_part,   label=label)
        model = tuner.hypermodel.build(best_hp)
        model.compile(metrics=custom_f1)
        model.fit(train_ds_part)
        my_dict['train_f1'] = model.evaluate(x = train_ds_part)[1]
        my_dict['val_f1']   = model.evaluate(x = val_ds_part)[1]
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
        num_trees=hp.Int('num_trees', min_value=10, max_value=300, step=25),
        max_depth=hp.Int('max_depth', min_value=2, max_value=25, step=1))
    model.compile(metrics=['accuracy'])
    return model


def create_bt_model(hp):
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=hp.Int('num_trees', min_value=10, max_value=300, step=25),
        max_depth=hp.Int('max_depth', min_value=2, max_value=25, step=1))
    model.compile(metrics=['accuracy'])
    return model


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
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

################################
#
# Training the basic random forest algorithm
#
################################
"""

tuner_rf = kt.RandomSearch(
    create_rf_model,
    objective=kt.Objective("val_f1", direction="max"),
    max_trials=10)

tuner_rf.search(train_ds,
                epochs=1,
                validation_data=val_ds
                )
best_model_rf = tuner_rf.get_best_models()[0]
print(best_model_rf)

perf_arr = data_size_tests(best_model_rf,df)
create_report(perf_arr, "reports/random_forest.png", "Random Forest Algorithm")
"""
#
# GradientBoostedTrees model
#



tuner_bt = kt.RandomSearch(
    create_bt_model,
    objective='val_accuracy',
    max_trials=2)

tuner_bt.search(train_ds,
                epochs=1,
                validation_data=val_ds
                )

best_hp = tuner_bt.get_best_hyperparameters()[0]
model = tfdf.keras.GradientBoostedTreesModel()
perf_arr = (tuner_bt, best_hp,df)
print("good")
create_report(perf_arr, "reports/gradien_boosting.png", "Gradient Boosted Trees Algorithm")


#
#   GradientBoostedTrees
#
"""
tuner = tfdf.tuner.RandomSearch(num_trials=20)
tuner.choice("min_examples", [2, 5, 7, 10])
tuner.choice("categorical_algorithm", ["CART", "RANDOM"])

local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [3, 4, 5, 6, 8])

global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

tuner.choice("use_hessian_gain", [True, False])
tuner.choice("shrinkage", [0.02, 0.05, 0.10, 0.15])
tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])

tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
tuned_model.fit(train_ds, verbose=2)

tuning_logs = tuned_model.make_inspector().tuning_logs()
print(tuning_logs.head())
print(tuning_logs[tuning_logs.best].iloc[0])

tuned_model.compile(["accuracy"])
tuned_test_accuracy = tuned_model.evaluate(val_ds, return_dict=True, verbose=0)["accuracy"]
#print(f"Test accuracy with the TF-DF hyper-parameter tuner: {tuned_test_accuracy:.4f}")

perf_arr = data_size_tests(df,tuner)
create_report(perf_arr, "reports/bt.png", "Gradient Boosted Trees Algorithm")
"""


#
#  Random Forest algorithm
#
"""

tuner = tfdf.tuner.RandomSearch(num_trials=20)
tuner.choice("num_trees", [10,30,50,70,90,110,130,150,170,190,210,230,250])
tuner.choice("max_depth", [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])


tuned_model = tfdf.keras.RandomForestModel(tuner=tuner)
tuned_model.fit(train_ds,
                verbose=2,
                objective='precision')

tuning_logs = tuned_model.make_inspector().tuning_logs()
print(tuning_logs.head())
print(tuning_logs[tuning_logs.best].iloc[0])

tuned_model.compile(["f1_score"])

#tuned_test_accuracy = tuned_model.evaluate(val_ds, return_dict=True, verbose=0)
#print(f"Test accuracy with the TF-DF hyper-parameter tuner: {tuned_test_accuracy:.4f}")

perf_arr = data_size_tests(df,tuner)
create_report(perf_arr, "reports/random_forest.png", "Random Forest Algorithm")
"""
"""
num_trees_arr =  np.arange(10,300,5)
max_depth_arr = np.arange(2,40,1)


num_iterations = 100
validation_acc = -10e10

for i in range(num_iterations):

    num_trees = np.random.choice(num_trees_arr)
    max_depth = np.random.choice(max_depth_arr)
    model = tfdf.keras.RandomForestModel(num_trees = num_trees.item(),
                                         max_depth = max_depth.item())
    model.fit(train_ds, verbose = 0)
    model.compile(metrics = custom_f1)
    print("Training f1   : ",   model.evaluate(train_ds)[1])
    print("Validation f1 : ", model.evaluate(val_ds)[1])

    if (validation_acc < model.evaluate(val_ds)[1]):
        validation_f1 = model.evaluate(val_ds)
        max_depth_best = max_depth
        num_trees_best = num_trees

print("BEST RANDOM FOREST MODEL")
print("max_depth:", max_depth_best)
print("num_trees:", num_trees_best)
print("f1 score: ", validation_f1[1])

best_model_rf = tfdf.keras.RandomForestModel(num_trees = num_trees_best.item(),
                                             max_depth =  max_depth_best.item())
perf_arr_rf = data_size_tests(best_model_rf, df)
create_report(perf_arr_rf, "reports/random_forest.png", "Random Forest Algorithm")

"""


#
# Gradient Boosted trees
#
"""
min_examples_arr           =  [2, 5, 7, 10]
max_depth_arr              =  [3, 4, 5, 6, 8]

num_iterations = 2
validation_acc = -10e10

for i in range(num_iterations):
    min_example          = random.choice(min_examples_arr)
    max_depth             =  random.choice(max_depth_arr)

    model = tfdf.keras.GradientBoostedTreesModel(min_examples                  = min_example,
                                                 max_depth                     = max_depth)
    model.fit(train_ds, verbose = 0)
    model.compile(metrics = custom_f1)
    print("Training f1   : ",   model.evaluate(train_ds)[1])
    print("Validation f1 : ", model.evaluate(val_ds)[1])

    if (validation_acc < model.evaluate(val_ds)[1]):
        validation_f1    = model.evaluate(val_ds)
        min_example_best = min_example
        max_depth_best   = max_depth

print("min example best", min_example_best)
model =  tfdf.keras.GradientBoostedTreesModel()
perf_arr_gb = data_size_tests(model, df, label)
create_report(perf_arr_gb, "reports/gradien_boosting.png", "Gradient Boosted Trees Algorithm")
"""