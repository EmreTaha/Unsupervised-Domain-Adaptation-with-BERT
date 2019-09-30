import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import re
import pandas as pd 
import gzip 
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split

def plotAndSave(target,source,name="accuracy"):
    # Visualizes loss or accuracy between target and source datasets

    # Create count of the number of epochs
    epoch_count = range(1, len(source) + 1)

    # Visualize loss history
    plt.plot(epoch_count, source, 'r--')
    plt.plot(epoch_count, target, 'b-')
    plt.legend(['Source ' + name, 'Target ' + name])
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.savefig(name+'.png')

''' Download, process and split the dataset'''

def split_d(dataset):
    dataset.columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'sentence',
           'sentiment', 'summary', 'unixReviewTime', 'reviewTime']
    dataset["sentiment"] = dataset["sentiment"].astype(np.int64)-1 
    def convert_to_three(value):
      if value == 0 or value == 1:
        return 0
      elif value == 2:
        return 1
      elif value == 3 or value == 4:
        return 2
    dataset["sentiment"] = dataset["sentiment"].map(convert_to_three)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    dp_train,dp_test = train_test_split(dataset,test_size=0.2)
    return dp_train,dp_test 

# Open the gz file and read each entry as dictionary
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
         yield eval(l) 

# Add the data entries to a pandas frame
def getDF(path):
    i = 0 
    df = {}
    for d in parse(path):
         df[i] = d 
         i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# Download the data and return its local path
def download_and_load_datasets(force_download=False,origin="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz"):
  ff = origin.split("/")[-1]
  dataset = tf.keras.utils.get_file(
      fname=ff,
      origin=origin, 
      extract=False)

  return dataset


def balance_dataset(db, label_column="sentiment"):
    # Balances  a pandaframe dataset according to given label_column
    db = db.groupby(label_column)
    db = db.apply(lambda x: x.sample(db.size().min()).reset_index(drop=True))
    db = db.sample(frac=1).reset_index(drop=True) #shuffle again