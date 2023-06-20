import torch
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class STSBertModel(torch.nn.Module):

    def __init__(self):

        super(STSBertModel, self).__init__()

        word_embedding_model = models.Transformer('sentence-transformers/all-distilroberta-v1', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def forward(self, input_data):

        output = self.sts_model(input_data)

        return output


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, dataset):

        similarity = [i['similarity_score'] for i in dataset]
        self.label = [i/5.0 for i in similarity]
        self.sentence_1 = [i['sentence1'] for i in dataset]
        self.sentence_2 = [i['sentence2'] for i in dataset]
        self.text_cat = [[str(x), str(y)] for x,y in zip(self.sentence_1, self.sentence_2)]

    def __len__(self):

        return len(self.text_cat)

    def get_batch_labels(self, idx):

        return torch.tensor(self.label[idx])

    def get_batch_texts(self, idx):

        return tokenizer(self.text_cat[idx], padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def collate_fn(texts):

  num_texts = len(texts['input_ids'])
  features = list()
  for i in range(num_texts):
      features.append({'input_ids':texts['input_ids'][i], 'attention_mask':texts['attention_mask'][i]})
  
  return features


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self,  loss_fct = torch.nn.MSELoss(), cos_score_transformation=torch.nn.Identity()):
      
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.cos = torch.nn.CosineSimilarity(dim=1)


    def forward(self, input, label):

        embedding_1 = torch.stack([inp[0] for inp in input])
        embedding_2 = torch.stack([inp[1] for inp in input])

        output = self.cos_score_transformation(self.cos(embedding_1, embedding_2))

        return self.loss_fct(output, label.squeeze())


