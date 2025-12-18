import os
import re
import pandas as pd

import nltk
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def loadRawData(PATH):
  '''
  Loads raw data from a string

  Input: PATH (str)
  Output: text (str)
  '''
  if type(PATH)!=str:
    raise TypeError('PATH must be a string!')
  if PATH=='':
    raise ValueError('PATH cannot be empty!')
  
  with open(PATH, 'r') as f:
    text = f.read()
  return text


def makeCleanTokens(s):
  '''
  Makes a list containing every word in a string, excluding stop words and "said"
  '''
  if type(s)!=str:
    raise TypeError('s must be a string!')
  if s=='':
    raise ValueError('s cannot be empty!')

  s=s.lower() # so words that are the same but with different capitalization won't be counted differently
  tokens=[]
  for word in re.findall(r"\b[a-z|0-9]+",s):
    # looking for occurrences of letters and numbers
    if word in stop_words or word=='said':
      # removing stop words from the pool
      # since Frank Herbert rarely uses verbs other than "said" in dialogue, I will remove that from the pool too
      continue
    tokens.append(word)
    # adding to a list of the words in the text
  return tokens


def separateAndTokenize(text):
  '''
  Separates Dune into chapters (skipping titles and appendices)
  and tokenizes each chapter (while removing stop words).

  Input: text (str)
  Output: separated (list), tokenized (list)
  '''
  if type(text)!=str:
    raise TypeError('text must be a string!')
  if text=='':
    raise ValueError('text cannot be empty!')

  separated=[] # list of chapter strings
  tokenized=[] # list of tokenized lists per chapter
  i=0
  for s in re.split('= = = = = =', text):
    if i in [0,23,39,51,52,53,54,55]: # skipping titles and appendices
      i+=1
      continue
    i+=1
    separated.append(s)
    tokenized.append(makeCleanTokens(s))

  return separated, tokenized


def makeCleanData(separated, tokenized):
  '''
  Makes DataFrame of separated chapters and clean tokens from those chapters

  Input: separated (list), tokenized (list)
  Output: cleanData (DataFrame)
  '''
  if type(separated)!=list or type(tokenized)!=list:
    raise TypeError('separated and tokenized must be lists!')

  cleanData = pd.DataFrame({
      'Chapter': list(range(1,49)),
      'Text': separated,
      'Clean Tokens': tokenized
  })

  return cleanData


def saveDataFrame(df, saveToPath):
  '''
  Saves a DataFrame to a CSV file

  Input: df (DataFrame), saveToPath (str)
  '''
  if type(df)!=pd.core.frame.DataFrame:
    raise TypeError('df must be a DataFrame!')
  if type(saveToPath)!=str:
    raise TypeError('saveToPath must be a string!')
  
  os.makedirs(os.path.dirname(saveToPath), exist_ok=True)
  df.to_csv(saveToPath, index=False)
