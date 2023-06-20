import json, pandas as pd
from tqdm import tqdm
import re
from sklearn.metrics.pairwise import cosine_distances as cos_dist
from collections import Counter
from itertools import chain
import numpy as np
from rouge import Rouge
from sentence_transformers import SentenceTransformer

test_data_path = '' # patentSBERTa_test_data.csv file path

test_column = 'summary' # claim1 or summary

section_list = ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]

if __name__ == "__main__" :

    result = pd.read_csv(test_data_path)

    # Rouge

    rouge = Rouge()
    scores = rouge.get_scores(result['summary'], result['abstract'], avg=True)
    print(scores)

    # PatentSBERTa
    sentences = result[test_column]

    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    embeddings = model.encode(sentences)

    # K = 10
    fold_len = 6
    av_acc = 0
    av_pre = 0
    av_rec = 0
    av_f1 = 0
    for fold in range(fold_len):
        distance = 'cosine'
        sl_idx = 5000

        idx = np.arange(6000)
        train_idx = np.random.choice(idx, size=sl_idx)
        test_idx = np.array(list(set(idx) - set(train_idx)))
        X = embeddings[train_idx]
        test_X = embeddings[test_idx]

        sec = []
        for i in tqdm(range(len(result))):
            sec.append(list(result.loc[i][section_list][result.loc[i][section_list] == 1].index))

        # del result["section"]
        result["section"] = sec

        result.to_csv("./test_set.csv", index=False)
        # sec_dict =dict(zip(range(len(section_list)), section_list))

        z = cos_dist(X, test_X)
        ag = z.argsort(axis=0)

        # params
        perform = {}
        K = 20
        # for K in range(3, 21):
        idx = ag[:K]

        neighbors = np.array(sec)[idx]

        ng_y_list = []
        for i in tqdm(range(1000)):
            ng_y_list.append(list(chain(*neighbors[:, i])))

        true_y = sec[4000:]
        pred_y = []
        for i in tqdm(range(1000)):
            c = Counter(ng_y_list[i])
            n_label = len(true_y[i])
            pred_y.append(list(np.array(c.most_common(n_label))[:, 0]))

        print(pred_y[:10])
        print(true_y[:10])

        ##### multi-label classification
        eval = dict(zip(section_list, [{'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0} for _ in range(len(section_list))]))
        for label in tqdm(eval.keys()):
            tp = 0
            fn = 0
            fp = 0
            tn = 0
            for i in range(len(pred_y)):
                if label in true_y[i]:
                    if label in pred_y[i]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if label in pred_y[i]:
                        fp += 1
                    else:
                        tn += 1
            eval[label]["TP"] = tp
            eval[label]["FN"] = fn
            eval[label]["FP"] = fp
            eval[label]["TN"] = tn

        confusion_matrix = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for label in eval.keys():
            tp += eval[label]["TP"]
            fn += eval[label]["FN"]
            fp += eval[label]["FP"]
            tn += eval[label]["TN"]
        confusion_matrix["TP"] = tp
        confusion_matrix["FN"] = fn
        confusion_matrix["FP"] = fp
        confusion_matrix["TN"] = tn

        #### micro average
        precision = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"])
        recall = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])
        accuracy = (confusion_matrix["TP"] + confusion_matrix["TN"]) / \
                   (confusion_matrix["TP"] + confusion_matrix["FN"] + confusion_matrix["FP"] + confusion_matrix["TN"])
        f1 = 2 * (precision * recall) / (precision + recall)

        print(accuracy, precision, recall, f1)
        # perform[K] = [accuracy, precision, recall, f1]
        av_pre += precision
        av_rec += recall
        av_acc += accuracy
        av_f1 += f1

    av_pre /= 6
    av_rec /= 6
    av_acc /= 6
    av_f1 /= 6
    print(av_acc, av_pre, av_rec, av_f1)