import re
import random
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from CPC_Class import cpc_class

import warnings
warnings.filterwarnings('ignore')

def parallel(data, func, n_cores=8):
  
  data_split = np.array_split(data, n_cores)
  pool = Pool(n_cores)
  data = pd.concat(pool.map(func, data_split))
  pool.close()
  pool.join()
  
  return data

def sampling(data):
  path = glob(r'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/data/*')
  cpc_path = path[4]

  with open(cpc_path, 'rb') as f:
    cpc = pickle.load(f)
  
  sections = cpc['section']

  data1 = data[['id', 'date', 'text']].copy()
  data = data.drop(['id','date', 'text'], axis=1)
  negative_list = random.sample(range(1, len(data.index)), int(0.25 * len(data.index)))
  labels = data.columns
  data = data.to_dict('index')
  for idx in tqdm(data1.index):
    if idx in negative_list:
      instance = data[idx]
      original_labels = [k for k in instance.keys() if instance[k] == 1]
      original_sections = [re.sub(r"[0-9]", "", k)[0] for k in instance.keys() if instance[k] == 1]
      candidate_labels = [k for k in instance.keys() if re.sub(r"[0-9]", "", k)[0] not in original_sections]
      candidate_labels = random.sample(candidate_labels, len(original_sections))
      for k in instance.keys():
        if k in original_labels:
          data[idx][k] = 0
        elif k in candidate_labels:
          data[idx][k] = 1
  
  data = pd.DataFrame(data).T
  data = pd.concat([data1, data], axis = 1)
  del data1

  return data

if __name__=='__main__':
  path = glob(r'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/data/*')
  train_path = path[2]
  random.seed(333)
  data = pd.read_csv(train_path)
  data = data.iloc[random.sample(range(1, len(data.index)), int(0.25 * len(data.index))),:]
  
  save_path = 'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/data'
  data.to_csv(save_path + '/df_sampled.csv', index=False)