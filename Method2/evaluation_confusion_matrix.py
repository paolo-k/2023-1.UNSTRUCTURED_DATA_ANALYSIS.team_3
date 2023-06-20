import re
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from CPC_Class import cpc_class
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


file_path = [
    'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/result',
    'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP'
]
with open(file_path[1] + '/Structured_CPC.pickle', 'rb') as f:
    cpc = pickle.load(f)

sections = {v.description:k for k, v in cpc['section'].items()} # 이후에 subclass단 추가 시, 손 봐야 함

### 모델 testing ###
original = pd.read_csv(file_path[0] + '/testing_description.csv')
labelled = pd.read_json(file_path[1] + '/result_evaluation.json').rename(columns={'LHS': 'text', 'RHS': 'label'})
labelled['id'] = pd.read_csv(file_path[0] + '/testing_columns.csv')['id']
labelled['label'] = [sections[i] for i in labelled['label']]
testset = pd.merge(original, labelled, how='right').drop('text', axis=1)
del original
del labelled

testset = testset.to_dict('records')

label_confusion = []
for i in range(int(len(testset)/len(sections))): # 여기도 이후에 subclass단 추가 시, 손 봐야 함
    instances = testset[i*len(sections):((i+1)*len(sections))]
    max_sim = 0
    label_list = [re.sub(r"[^a-zA-Z]", '', j)[0] for j in instances[0]['cpc_code'].split(',')]
    while instances:
        instance = instances.pop()
        if instance['similarity'] > max_sim:
            max_sim = instance['similarity']
            top = instance['label']

    pred_list = {section : 1 if section == top else 0 for section in sections.values()}
    for section, pred in pred_list.items():
        if pred == 0:
            label_confusion.append({
                section in label_list: 'FN',
                section not in label_list: 'TN'
            }.get(True))
        else:
            label_confusion.append({
                section in label_list: 'TP',
                section not in label_list: 'FP'
            }.get(True))


print(Counter(label_confusion))

confusion_vector=[
    [1 if i == 'TP' or i == 'FN' else 0 for i in label_confusion],
    [1 if i == 'TP' or i == 'FP' else 0 for i in label_confusion]
]

y_true = confusion_vector[0]
y_pred = confusion_vector[1]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

p = precision_score(y_true, y_pred)
print(p)
r = recall_score(y_true, y_pred)
print(r)
f1 = f1_score(y_true, y_pred)
print(f1)


### claim1 vs summary ###

for name in ['claim1', 'summary']:
    original = pd.read_csv(file_path[0] + f'/description_{name}.csv').drop('Unnamed: 0', axis=1)
    labelled = pd.read_json(file_path[1] + f'/{name}_evaluation.json').rename(columns={'LHS': 'text', 'RHS': 'label'})
    labelled['label'] = [sections[i] for i in labelled['label']]
    testset = pd.concat([original, labelled], axis=1).drop('text', axis=1)
    del original
    del labelled

    testset = testset.to_dict('records')

    label_confusion = []
    for i in range(int(len(testset)/len(sections))): # 여기도 이후에 subclass단 추가 시, 손 봐야 함
        instances = testset[i*len(sections):((i+1)*len(sections))]
        max_sim = 0
        label_list = [i['cpc_code'] for i in instances if i['similarity_score'] > 0.1]
        while instances:
            instance = instances.pop()
            if instance['similarity'] > max_sim:
                max_sim = instance['similarity']
                top = instance['label']

        pred_list = {section : 1 if section == top else 0 for section in sections.values()}
        for section, pred in pred_list.items():
            if pred == 0:
                label_confusion.append({
                    section in label_list: 'FN',
                    section not in label_list: 'TN'
                }.get(True))
            else:
                label_confusion.append({
                    section in label_list: 'TP',
                    section not in label_list: 'FP'
                }.get(True))

    print(Counter(label_confusion))

    confusion_vector=[
        [1 if i == 'TP' or i == 'FN' else 0 for i in label_confusion],
        [1 if i == 'TP' or i == 'FP' else 0 for i in label_confusion]
    ]

    y_true = confusion_vector[0]
    y_pred = confusion_vector[1]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    p = precision_score(y_true, y_pred)
    print(p)
    r = recall_score(y_true, y_pred)
    print(r)
    f1 = f1_score(y_true, y_pred)
    print(f1)