import os
import pickle
import numpy as np
import pandas as pd
import torch
import re
import torch.multiprocessing as mp
import pyarrow as pa
import bert_similarity


from glob import glob
import multiprocess
from multiprocessing import Pool
from transformers import BertTokenizer
from datasets import Dataset
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from CPC_Class import cpc_class

def load_data():

  '''
  토크나이저, test set, CPC 코드 로드
  :return: 토크나이저, 데이터셋 저장경로, CPC 코드 객체, test set
  '''

  print('load dataset...')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  path = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/*')
  test_path = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/*test*')[0]
  cpc_path = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/*.pickle')[0]

  with open(cpc_path, 'rb') as f:
    cpc = pickle.load(f)
  test = pd.read_csv(test_path)

  return tokenizer, path, cpc, test


def load_data_summary():

  '''
  Method 1, Method 2 교호작용 확인용 test set 로드
  :return: 토크나이저, 경로, CPC 코드 객체, negative label set, train set, test set
  '''

  print('load dataset...')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  path = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/*')
  test_path2 = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/test_set.csv')[0]
  cpc_path = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/*.pickle')[0] # pickle path

  with open(cpc_path, 'rb') as f:
    cpc = pickle.load(f)

  test2 = pd.read_csv(test_path2)
  test2['id'] = test2.index
  test_claim1 = test2[['id', 'claim1', 'cpc_ids']].rename(columns = {'id':'id', 'claim1':'text', 'cpc_ids':'cpc_code'})
  test_summary = test2[['id', 'summary', 'cpc_ids']].rename(columns = {'id':'id', 'summary':'text', 'cpc_ids':'cpc_code'})

  return tokenizer, path, cpc, test_claim1, test_summary


def parallel(data, func, n_cores=8):
  '''
  pd.DataFrame 기반으로 병렬처리
  :param data: 데이터셋(DataFrame)
  :param func: 함수
  :param n_cores: worker 개수
  :return: 병렬처리 완료된 데이터
  '''

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
  path = glob(r'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/*.pickle')
  cpc_path = path[0]

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


def BERT_preprocess1(data, name = str(), n=0):
  '''
  특허 labelling
  :param data: 전처리된 data(cpc 코드 list화)
  :param n: negative set 여부(0: positive set, 1: negative set)
  :return: labelling된 특허 data
  '''

  print('labelling patents...')

  save_path = 'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/result/'

  try:
    data = parallel(data, preprocess)
  except:
    print('this data is already preprocessd')

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


def BERT_preprocess2(data, name = str()):

  '''
  :param data: labelled patent data
  :return: formatted patent data
  '''

  print('transform dataset pd to torch dataset...')

  dataset = Dataset(pa.Table.from_pandas(data))

  save = f'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/{name}_datasets.json'
  dataset.to_json(save)

  return dataset


def model_load():
  '''
  fine-tuning 완료된 baseline model 로드
  :return: baseline model
  '''

  path = r'C:\Users\kw764\Dropbox\강웅\02 data\03 homework\04 NLP'
  model = torch.load(path + '\\BERT_baseline.pt',map_location='cuda:0')

  return model


def predict_sts(tokenizer, trained_model, texts):

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")

  trained_model.to(device)
  trained_model.eval()

  test_input = tokenizer(texts, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
  test_input.to(device)
  test_input['input_ids'] = test_input['input_ids']
  test_input['attention_mask'] = test_input['attention_mask']
  del test_input['token_type_ids']

  test_output = trained_model(test_input)['sentence_embedding']
  sim = torch.nn.functional.cosine_similarity(test_output[0], test_output[1], dim=0).item()

  return sim


def evaluate(trained_model, test_dataset):

  print('evaluating')

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

  sentence_1_test = [i['sentence1'] for i in test_dataset]
  sentence_2_test = [i['sentence2'] for i in test_dataset]
  text_cat_test = [[str(x), str(y)] for x, y in zip(sentence_1_test, sentence_2_test)]

  result_evaluation = [
    predict_sts(tokenizer, trained_model, i)
    for i in tqdm(text_cat_test)
  ]
  result_evaluation = pd.DataFrame({
    'LHS':sentence_1_test,
    'RHS':sentence_2_test,
    'similarity': result_evaluation
  })

  return result_evaluation


def interaction_testing(test_claim1, test_summary, cpc):
  '''
  Claim1 vs Summarization 성능 비교를 통한 Method 1, Method 2 교호작용 확인
  :param test_claim1: Claim 1 testset
  :param test_summary: Summarization testset
  :param cpc: CPC 코드 객체
  :return: void
  '''

  claim1_data = BERT_preprocess1(test_claim1, cpc, 'claim1')
  claim1_dataset = BERT_preprocess2(claim1_data, 'claim1')

  claim1_evaluation = evaluate(trained_model, claim1_dataset)
  claim1_evaluation.to_json(save_path + '/claim1_evaluation.json')

  summary_data = BERT_preprocess1(test_summary, cpc, 'summary')
  summary_dataset = BERT_preprocess2(summary_data, 'summary')

  summary_evaluation = evaluate(trained_model, summary_dataset)
  summary_evaluation.to_json(save_path + '/summary_evaluation.json')



if __name__ == '__main__':

  # Method 2 Testing
  save_path = 'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP'

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")

  tokenizer, path, cpc, test = load_data()
  trained_model = model_load()

  test_data = BERT_preprocess1(test, cpc)
  test_dataset = BERT_preprocess2(test_data)

  result_evaluation = evaluate(trained_model, test_dataset)
  result_evaluation.to_json(save_path + '/result_evaluation.json')


  # Method 1, Method 2 interaction testing
  tokenizer, path, cpc, test_claim1, test_summary = load_data_summary()

  interaction_testing(test_claim1, test_summary, cpc)