import pickle

class cpc_class :
  def __init__(self, symbol, description):
    self.symbol = symbol
    self.description = description
    self.parent = []
    self.child = []
    self.level = None

  if __name__ == "__main__":
    path = 'C:/Users/kw764/Dropbox/강웅/02 data/03 homework/04 NLP/Structured_CPC.pickle' # pickle path

    with open(path, 'rb') as f:
      data = pickle.load(f)
