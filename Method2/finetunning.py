import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pyarrow as pa
import bert_similarity



from glob import glob
import multiprocess
from multiprocessing import Pool
from transformers import BertTokenizer
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from CPC_Class import cpc_class


def load_data():

  '''
  토크나이저, 특허 데이터, CPC 코드 로딩
  :return: 토크나이저, 경로, CPC 코드 객체, negative label set, train set, test set
  '''

  print('load dataset...')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  path = glob(r'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/data/*')
  test_path = path[1]
  train_path = path[3]
  cpc_path = path[4]

  with open(cpc_path, 'rb') as f:
    cpc = pickle.load(f)

  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)

  return tokenizer, path, cpc, train, test


def parallel(data, func, n_cores=8):


  data_split = np.array_split(data, n_cores)
  pool = Pool(n_cores)
  data = pd.concat(pool.map(func, data_split))
  pool.close()
  pool.join()

  return data


def preprocess(data):

  '''
  기본적인 전처리 코드 : cpc 코드를 행렬 형태에서 list로 변환
  :param data: raw data
  :return: 전처리된 data
  '''

  print('preprocessing...')

  idx = data['id'].tolist()
  data1 = data['text'].tolist()
  data = data.drop(['id', 'date', 'text'], axis=1).T.to_dict()
  cpc_code = []
  while data:
    print(f'remain : {len(data)}')
    patent = data.pop(list(data.keys())[0])
    cpc_code.append([i for i in patent.keys() if patent[i] >= 1])
  data = pd.DataFrame(
    {
      'id':idx,
      'text':data1,
      'cpc_code':cpc_code
    }
  )

  return data


def extract_cpc_description(idx, patent, cpc):

  '''
  cpc 객체로부터 description 추출
  :param idx: 특허 id
  :param patent: 특허 클레임
  :param cpc: 특허 cpc 코드
  :param n: negative set 여부(0: positive set, 1: negative set)
  :return: cpc description
  '''

  print('extract cpc...')

  codes = []
  descriptions = []
  similarity_score = []
  sections = cpc['section']
  if type(patent['cpc_code']) == str:
    patent['cpc_code'] = [re.sub(r"[^0-9A-Za-z]", "", i) for i in patent['cpc_code'].split(',')]

  original_section = []
  for i in patent['cpc_code']:

    original_section.append(re.sub(r"[0-9]", "", i)[0])

  for sect in sections.keys():

    codes.append(sect)
    descriptions.append(cpc['section'][sect].description)
    similarity_score.append({
      sect in original_section:1.0,
      sect not in original_section:0.0
    }.get(True))
      
  if len(descriptions) > 0:
    description = pd.DataFrame({
      'id' : [idx for i in range(len(codes))],
      'cpc_code' : codes,
      'description' : descriptions,
      'similarity_score' : similarity_score
    })

    return description


def extractor(data):
  path = glob(r'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/data/*')
  cpc_path = path[4]

  with open(cpc_path, 'rb') as f:
    cpc = pickle.load(f)

  data = data.set_index('id')
  data = data.T.to_dict()
  description = pd.DataFrame()
  while data:
    idx = list(data.keys())[0]
    patent = data.pop(idx)
    description = pd.concat([
      description,
      extract_cpc_description(idx, patent, cpc)
    ])
    description = description.reset_index().drop('index', axis=1)

  return description


def BERT_preprocess1(data, name = str()):
  '''
  특허 labelling
  :param data: 전처리된 data(cpc 코드 list화)
  :return: labelling된 특허 data
  '''

  print('labelling patents...')

  save_path = 'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/result/'

  data = parallel(data, preprocess)
  data1 = data[['id', 'text']].copy()

  save = save_path + f'preprocessed_{name}.csv'
  data.to_csv(save)

  description = parallel(data, extractor)

  save = save_path + f'description_{name}.csv'
  description.to_csv(save)

  data = pd.merge(data1, description, how='right')
  data = data.iloc[:, [1, 3, 4]].rename(columns={'text': 'sentence1', 'description': 'sentence2', 'similarity_score':'similarity_score'})

  save = save_path + f'preprocess_completed_{name}.csv'
  data.to_csv(save)

  return data


def BERT_preprocess2(data):

  '''
  huggingface dataset으로 변환
  :param data: labelled patent data
  :return: huggingface dataset
  '''

  print('transform dataset pd to hg...')

  dataset = Dataset(pa.Table.from_pandas(data))
  
  save = 'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/result/hgdatasets.json'
  dataset.to_json(save)

  return dataset


def model_train(dataset, save_path, epochs, learning_rate, bs):

  '''
  BERT fine tuning
  :param dataset: huggingface dataset
  :param epochs: epoch
  :param learning_rate: learning rate
  :param bs: batch size
  :return: fine tuned BERT model
  '''

  print('training BERT...')

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:1" if use_cuda else "cpu")

  print('==========')

  model = bert_similarity.STSBertModel()
  
  criterion = bert_similarity.CosineSimilarityLoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)
  
  print('==========')

  train_dataset = bert_similarity.DataSequence(dataset)
  train_dataloader = DataLoader(train_dataset, num_workers=16, batch_size=bs, shuffle=True)

  if use_cuda:
    model.to(device)
    criterion = criterion.to(device)

  best_acc = 0.0
  best_loss = 1000

  for i in range(epochs):

    total_acc_train = 0
    total_loss_train = 0.0
    
    for train_data, train_label in tqdm(train_dataloader, desc='training'):
      train_data = train_data.to(device)
      train_label = train_label.to(device)
      train_data['input_ids'] = train_data['input_ids']
      train_data['attention_mask'] = train_data['attention_mask']
      del train_data['token_type_ids']

      train_data = bert_similarity.collate_fn(train_data)

      output = [model(feature)['sentence_embedding'] for feature in train_data]

      loss = criterion(output, train_label)
      total_loss_train += loss.item()

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    print(f'Epochs: {i + 1} | Loss: {total_loss_train / len(dataset): .3f}')
    model.train()

    torch.save(model, save_path + '/BERT_baseline.pt')

  return model


def fine_tunning(train, save_path, epochs, learning_rate, bs):

  '''
  전처리부터 fine tuning까지 종합
  :param train_negative: negative label set
  :param train: train set
  :param epochs: epoch
  :param learning_rate: learning rate
  :param bs: batch size
  :return: fine tuned BERT model
  '''

  print('start fine tuning...')
  
  train = BERT_preprocess1(train, 'train')
  train = BERT_preprocess2(train)
  train = train.remove_columns(["Unnamed: 0"])
  model = model_train(train['train'], save_path, epochs, learning_rate, bs)

  return model



if __name__ == '__main__':
  EPOCHS = 4
  LEARNING_RATE = 3e-5
  BATCH_SIZE = 128
  save_path = 'C:/Users/SAVANNA_WS_02/Documents/KW/NLP/result'
  
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"
  
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

  tokenizer, path, cpc, train = load_data()
  trained_model = fine_tunning(train, save_path, EPOCHS, LEARNING_RATE, BATCH_SIZE)
