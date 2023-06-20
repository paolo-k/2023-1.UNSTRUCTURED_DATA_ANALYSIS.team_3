import torch
import re
import json
import pickle

# Importing the LongT5 modules from huggingface/transformers
from transformers import AutoTokenizer
from transformers import LongT5Model
from transformers import LongT5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from transformers import BertForSequenceClassification

# Importing libraries
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

from rouge import Rouge

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
  """display dataframe in ASCII format"""

  console = Console()
  table = Table(
    Column("source_text", justify="center"),
    Column("target_text", justify="center"),
    title="Sample Data",
    pad_edge=False,
    box=box.ASCII,
  )

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)


# training logger to log training progress
training_logger = Table(
  Column("Epoch", justify="center"),
  Column("Steps", justify="center"),
  Column("Loss", justify="center"),
  title="Training Status",
  pad_edge=False,
  box=box.ASCII,
)

class YourDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and
  loading it into the dataloader to pass it to the
  neural network for finetuning the model

  """

  def __init__(
      self, dataframe, tokenizer, source_len, target_len, source_text, target_text
  ):
    """
    Initializes a Dataset class

    Args:
        dataframe (pandas.DataFrame): Input dataframe
        tokenizer (transformers.tokenizer): Transformers tokenizer
        source_len (int): Max length of source text
        target_len (int): Max length of target text
        source_text (str): column name of source text
        target_text (str): column name of target text
    """
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    """returns the length of dataframe"""

    return len(self.target_text)

  def __getitem__(self, index):
    """return the input ids, attention masks and target ids"""

    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    # cleaning data so as to ensure data is in string type
    source_text = " ".join(source_text.split())
    target_text = " ".join(target_text.split())

    source = self.tokenizer.batch_encode_plus(
      [source_text],
      max_length=self.source_len,
      pad_to_max_length=True,
      truncation=True,
      padding="max_length",
      return_tensors="pt",
    )
    target = self.tokenizer.batch_encode_plus(
      [target_text],
      max_length=self.summ_len,
      pad_to_max_length=True,
      truncation=True,
      padding="max_length",
      return_tensors="pt",
    )

    source_ids = source["input_ids"].squeeze()
    source_mask = source["attention_mask"].squeeze()
    target_ids = target["input_ids"].squeeze()
    target_mask = target["attention_mask"].squeeze()

    return {
      "source_ids": source_ids.to(dtype=torch.long),
      "source_mask": source_mask.to(dtype=torch.long),
      "target_ids": target_ids.to(dtype=torch.long),
      "target_ids_y": target_ids.to(dtype=torch.long),
    }


def train(num_epochs, tokenizer, model, device, train_loader, optimizer, val_loader, validate_every):
  """
  Function to be called for training with the parameters passed from main function
  """
  total_step = 0

  for epoch in tqdm(range(model_params["TRAIN_EPOCHS"])):
    model.train()

    total_loss = 0.0
    for _, batch in enumerate(train_loader, 0):
      y_ids = batch["target_ids"].to(device, dtype=torch.long)
      y_ids = y_ids[:, :-1].contiguous()

      ids = batch["source_ids"].to(device, dtype=torch.long)
      mask = batch["source_mask"].to(device, dtype=torch.long)
      optimizer.zero_grad()

      with torch.autocast("cuda"):
        outputs = model(
          input_ids=ids,
          attention_mask=mask,
          decoder_input_ids=y_ids,
        )
        loss = outputs[0]

      loss.mean().backward()
      optimizer.step()
      total_loss += loss.mean().item()
      total_step += 1

      if total_step % validate_every == 0:
        training_logger.add_row(str(epoch), str(total_step), str(loss))
        console.print(training_logger)
        average_loss = total_loss / validate_every
        print(
          f"Epoch [{epoch + 1}/{num_epochs}], Step [{total_step}/{len(train_loader)}], Training Loss: {average_loss:.4f}")

        # Validate the model
        val_ruoge = validate(tokenizer, model, device, val_loader)
        print(
          f"Epoch [{epoch + 1}/{num_epochs}], Step [{total_step}/{len(train_loader)}], Validation rouge: {val_ruoge:.4f}")
        print()

        total_loss = 0.0

      if total_step % 200 == 0 :
        console.log(f"[Saving Model]... Epoch{epoch} - {total_step}\n")
        # Saving the model after training
        path = os.path.join(output_dir, "model_files")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    console.log(f"[Saving Model]...Epoch{epoch} - Finished\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def validate(tokenizer, model, device, loader):
  """
  Function to evaluate model for predictions

  """
  # float32 -> float16
  model.half()
  model.eval()
  predictions = []
  actuals = []
  print(f"Validation Started")
  with torch.no_grad():
    for _, batch in tqdm(enumerate(loader, 0)):
      y = batch['target_ids'].to(device, dtype=torch.long)
      ids = batch['source_ids'].to(device, dtype=torch.long)
      mask = batch['source_mask'].to(device, dtype=torch.long)

      generated_ids = model.module.generate(
        input_ids=ids,
        attention_mask=mask,
        max_length=512,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
      )
      preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
      target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

      predictions.extend(preds)
      actuals.extend(target)

  rouge = Rouge()
  results = rouge.get_scores(predictions, actuals, avg=True)

  console.save_text(os.path.join(output_dir, "logs.txt"))

  console.log(f"[Validation Completed.]\n")
  console.print(
    f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
  )
  console.print(
    f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n"""
  )
  console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")
  console.print(results)

  return results


def test(tokenizer, model, test_dataset, device):
  """
  Function to evaluate model for predictions

  """

  count = 0

  # Creating the Training and Validation dataset for further creation of Dataloader
  test_set = YourDataSetClass(
    test_dataset,
    tokenizer,
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
  )

  del test_dataset

  # Defining the parameters for creation of dataloaders
  test_params = {
    "batch_size": model_params["VALID_BATCH_SIZE"],
    "shuffle": False,
    "num_workers": 0,
  }

  # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
  test_loader = DataLoader(test_set, **test_params)

  del test_set

  # float32 -> float16
  model.half()
  model.eval()
  predictions = []
  actuals = []
  count = 1

  with torch.no_grad():
    for _, batch in tqdm(enumerate(test_loader, 0)):
      y = batch['target_ids'].to(device, dtype=torch.long)
      ids = batch['source_ids'].to(device, dtype=torch.long)
      mask = batch['source_mask'].to(device, dtype=torch.long)

      generated_ids = model.module.generate(
        input_ids=ids,
        attention_mask=mask,
        max_length=255,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
      )
      preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
      # target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

      predictions.extend(preds)
      # actuals.extend(target)

      if _ % 500 == 0:
        with open(f'./outputs/data'+ str(count) +'.pickle', 'wb') as f:
          pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
        predictions = []
        count += 1

  # return predictions, actuals


def T5Trainer(model, tokenizer, train_dataset, val_dataset, source_text, target_text, model_params):
  """
  LongT5 trainer

  """

  # logging
  console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

  console.print(f"TRAIN Dataset: {train_dataset.shape}")
  console.print(f"TEST Dataset: {val_dataset.shape}\n")

  # Creating the Training and Validation dataset for further creation of Dataloader
  training_set = YourDataSetClass(
    train_dataset,
    tokenizer,
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
  )
  val_set = YourDataSetClass(
    val_dataset,
    tokenizer,
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
  )

  del train_dataset
  del val_dataset

  # Defining the parameters for creation of dataloaders
  train_params = {
    "batch_size": model_params["TRAIN_BATCH_SIZE"],
    "shuffle": True,
    "num_workers": 0,
  }

  val_params = {
    "batch_size": model_params["VALID_BATCH_SIZE"],
    "shuffle": False,
    "num_workers": 0,
  }

  # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
  training_loader = DataLoader(training_set, **train_params)
  val_loader = DataLoader(val_set, **val_params)

  del training_set
  del val_set

  # Defining the optimizer that will be used to tune the weights of the network in the training session.
  optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=model_params["LEARNING_RATE"]
  )

  # Training loop
  console.log(f"[Initiating Fine Tuning]...\n")

  train(model_params["TRAIN_EPOCHS"], tokenizer, model, device, training_loader, optimizer, val_loader, validate_every)

  # # evaluating test dataset
  # console.log(f"[Initiating Validation]...\n")
  # for epoch in range(model_params["VAL_EPOCHS"]):
  #   predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
  #   final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
  #   final_df.to_csv(os.path.join(output_dir, "predictions.csv"))
  #
  # console.save_text(os.path.join(output_dir, "logs.txt"))
  #
  # console.log(f"[Validation Completed.]\n")
  # console.print(
  #   f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
  # )
  # console.print(
  #   f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n"""
  # )
  # console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")
  #
  # rouge = Rouge()
  # results = rouge.get_scores(final_df["Generated Text"], final_df["Actual Text"], avg=True)
  # console.print(results)


##########################################################
#                                                        #
#                       execution                        #
#                                                        #
##########################################################

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512' ### 128, 256, 512 ....
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '1'

model_params = {
  # "MODEL" : "bert-base-uncased",
  "MODEL" : "pszemraj/long-t5-tglobal-base-16384-book-summary",
  # "MODEL": "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps",
  ### 가능하면 tglobal로 실험해주시기 바랍니다 ...
  ## tglobal model의 base, large, xl모델을 각각 사용하여 성능비교를 표로 만드는게 어떠한지...?

  ### LongT5 논문에서의 배치사이즈는 128로 고정입니다. 모델사이즈와 데이터가 크기에 작게 해봤습니다.
  "TRAIN_BATCH_SIZE": 1,  ## training batch size
  "VALID_BATCH_SIZE": 2,  ## validation batch size

  "TRAIN_EPOCHS": 10,  # number of training epochs
  "VAL_EPOCHS": 1,  # number of validation epochs

  "LEARNING_RATE": 0.001,  ### From LongT5 논문: For fine-tuning, we use a constant learning rate of 0.001

  "MAX_SOURCE_TEXT_LENGTH": 16380,  ### From fugging face: handling long input sequences (up to 16,384 tokens)
  "MAX_TARGET_TEXT_LENGTH": 510,  ### PatentSBERTa 논문에서 Input sequence를 510으로 고정하였으므로 이를 위해서 똑같이 맞춤

  "SEED": 42,  # set seed for reproducibility
}
# Importing the raw dataset

# file_name = "testing_datasets.json"
output_dir = "./outputs"

if __name__ == "__main__" :

  validate_every = 50

  # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
  # Further this model is sent to device (GPU/TPU) for using the hardware.
  # Setting up the device for GPU usage

  os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
  model = LongT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])

  # cond1 = re.compile('\d\d')
  # cond2 = re.compile('final_layer')
  #
  # decoder_update_index = 11
  #
  # print("Freezing BLOCKS")
  # # for name, param in model.named_parameters():
  # #   param.requires_grad = False
  #
  # for name, param in model.named_parameters():
  #   result1 = cond1.search(name)
  #   result2 = cond2.search(name)
  #   if result1 :
  #     if int(result1.group()) < decoder_update_index:
  #       print(name)
  #       param.requires_grad = False
  #   elif not result2 :
  #     print(name)
  #     param.requires_grad = False

  if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using DataParallel...")
    model = torch.nn.DataParallel(model, device_ids=[0,1], output_device=1)
  model.to(device)

  source_text = "claim_text"
  target_text = "abstract"

  # logging
  console.log(f"[Data]: Reading data...\n")

  # data = pd.read_csv(path + '/' + file_name)
  # data = data[[source_text, target_text]]

  # import json, pandas as pd
  # from tqdm import tqdm
  #
  # json_data = []
  # with open(path + '/' + "testing_datasets.json") as f:
  #   for line in f:
  #     json_data.append(json.loads(line)["sentence1"])

  path = "./data"
  file_name = "test_data.csv"
  test_data = pd.read_csv(path + '/' + file_name)
  test_data = test_data.reset_index(drop=True)
  test_data = test_data[["claim_text", "abstract"]]
  test_data["summary"] = ""
  # train_size = 0.9  ### LongT5 논문에서 train set을 90%로 사용함
  # val_size = 0.7
  # train_dataset = data.sample(frac=train_size, random_state=model_params["SEED"])
  # res_dataset = data.drop(train_dataset.index).reset_index(drop=True)
  # val_dataset = res_dataset.sample(frac=val_size, random_state=model_params["SEED"])
  # test_data = res_dataset.drop(val_dataset.index).reset_index(drop=True)
  # train_dataset = train_dataset.reset_index(drop=True)
  #

  ### tokenzier for encoding the text
  tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])

  # Set random seeds and deterministic pytorch for reproducibility
  torch.manual_seed(model_params["SEED"])  # pytorch random seed
  np.random.seed(model_params["SEED"])  # numpy random seed
  torch.backends.cudnn.deterministic = True

  print(torch.cuda.mem_get_info(torch.cuda.current_device()))

  test(tokenizer, model, test_data, device)

