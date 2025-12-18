import pandas as pd
import plotly.express as px

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

from dataProcessing import separateAndTokenize


def countWords(tokens):
  '''
  Counts words in a list of tokens, returning a dict of the words and their counts

  Input: tokens (list)
  Output: wordCounts (dict)
  '''
  if type(tokens)!=list:
    raise TypeError('tokens must be a list!')
  if tokens==[]:
    raise ValueError('tokens cannot be empty!')

  wordCounts={}

  for word in tokens:
    if word in wordCounts:
      wordCounts[word]+=1 # increments word in dict
    else:
      wordCounts[word]=1 # puts word in dict

  return wordCounts # {'word': count, ...}


def sortDict(d):
  '''
  Sorts a dict that has integer values.

  Input: d (dict)
  Output: sorted (dict)
  '''
  if type(d)!= dict:
    raise TypeError('d must be a dict!')
  if d=={}:
    raise ValueError('d cannot be empty!')

  sortList=[] # temporary list, will only contain words
  for word, n in d.items():
    if type(n)!=int: # extra error check, down here for efficiency
      raise TypeError('d must only have integer values!')

    if len(sortList)==0: # first word in sortList can't be sorted
      sortList.append(word)
      continue

    elif len(sortList)==1: # second word needs special sorting
      if d[sortList[0]]>n: # if second word has lower count than first
        sortList.append(word)
        continue
      else: # if second word has higher count than first
        sortList.insert(0,word)
        continue

    for j in range(len(sortList)): # sortList has more than 2+ words in it now
      if d[word]>d[sortList[j]]:
        sortList.insert(j,word)
        break

    if word not in sortList: # after everything, if word isn't in sorted, it is appended
      sortList.append(word)

  sorted={} # final result
  for word in sortList:
    sorted[word]=d[word] # assigns words in sortList their associated counts

  return sorted


def chWordCounts(tokenized):
  '''
  Counts and sorts words from a list of lists of tokens in each chapter.
  Returns a list containing the sorted word counts for each chapter.

  Input: tokenized (list)
  Output: chCounter (list)
  '''
  if type(tokenized)!=list:
    raise TypeError('tokenized must be a list!')
  if tokenized==[]:
    raise ValueError('tokenized cannot be empty!')

  chCounter=[] # will contain 48 sorted dicts

  for ch in tokenized:
    d = sortDict(countWords(ch))
    chCounter.append(d)

  return chCounter


def top5Chapter(chCounter):
  '''
  Separates the top 5 word entries per chapter.
  Returns two lists: one containing each chapter's top 5 words,
    and the other containing the associated counts.

  Input: chCounter (list)
  Output: chTop5Words (list), chTop5Nums (list)
  '''
  if type(chCounter)!=list:
    raise TypeError('chCounter must be a list!')
  if chCounter==[]:
    raise ValueError('chCounter cannot be empty!')

  chTop5Words=[] # top 5 words per chapter
  chTop5Nums=[] # associated counts
  # separate lists for data organization later

  for ch in chCounter:
    words=[] # this chapter's top 5
    nums=[] # this chapter's counts

    for i in range(5):
      words.append(list(ch.keys())[i])
      nums.append(list(ch.values())[i])

    chTop5Words.append(words) # append this chapter's data
    chTop5Nums.append(nums)

  return chTop5Words, chTop5Nums


def bookWordCounts(chCounter):
  '''
  Combines word counts per chapter into a sorted total word count.

  Input: chCounter (list)
  Output: totalCount (dict)
  '''
  if type(chCounter)!=list:
    raise TypeError('chCounter must be a list!')
  if chCounter==[]:
    raise ValueError('chCounter cannot be empty!')

  totalCount={}

  for ch in chCounter:
    for word, count in ch.items():
      if word in totalCount.keys():
        totalCount[word]+=count # combines counts of duplicate words
      else:
        totalCount[word]=count # appends new words
  totalCount=sortDict(totalCount) # sorts totalCount

  return totalCount


def bookTopX(totalCount, x=5):
  '''
  Finds the top x words across the text. Default value of x is 5.

  Input: totalCount (dict), x (int)
  Output: totalTopX (dict)
  '''
  if type(totalCount)!=dict:
    raise TypeError('totalCount must be a dict!')
  if totalCount=={}:
    raise ValueError('totalCount cannot be empty!')
  if type(x)!=int:
    raise TypeError('x must be an int!')
  if x<1:
    raise ValueError('x must be greater than 0!')
  if x>len(totalCount.keys()):
    raise ValueError('x is too high!')

  totalTopX={} # unlike in top5Chapter, a dict can be used for simplicity's sake

  for word, count in totalCount.items():
    totalTopX[word]=count
    if len(totalTopX.keys())>=x: # stop if there are x keys
      break

  return totalTopX


def findUniqueWords(totalCount):
  '''
  Finds words used only once in a text.

  Input: totalCount (dict)
  Output: uniqueWords (list)
  '''
  if type(totalCount)!=dict:
    raise TypeError('totalCount must be a dict!')
  if totalCount=={}:
    raise ValueError('totalCount cannot be empty!')

  uniqueWords=[] # uniqueWords is a list because they all have count 1

  for word, count in totalCount.items():
    if count==1:
      uniqueWords.append(word)

  return uniqueWords


def findUniqueTop5(chCounter):
  '''
  Finds words that appear in only one chapter's top 5.
  Returns a sorted dict of top word appearances and a
  list of words that only reach the top 5 once.

  Input: chCounter (list)
  Output: countTop5 (dict), uniqueTop5 (list)
  '''

  chTop5Words, chTop5Nums=top5Chapter(chCounter)

  countTop5={}

  for i in range(48):
    for j in range(5):
      if chTop5Words[i][j] in countTop5.keys():
        countTop5[chTop5Words[i][j]]+=1
      else:
        countTop5[chTop5Words[i][j]]=1
  countTop5=sortDict(countTop5)

  uniqueTop5=[]

  for word, count in countTop5.items():
    if count == 1:
      uniqueTop5.append(word)

  return countTop5, uniqueTop5


def makeChSentiments(separated):
  '''
  Calculates the sentiments of each chapter of a text

  Input: separated (list)
  Output: chSentiments (list)
  '''
  if type(separated)!=list:
    raise TypeError('separated must be a list!')
  if separated==[]:
    raise ValueError('separated cannot be empty!')

  chSentiments=[]

  for ch in separated:
    sent=sid.polarity_scores(ch)['compound'] # takes the compound sentiment of each chapter
    chSentiments.append(sent)

  return chSentiments


def makeDuneData(text):
  '''
  Uses functions to take the raw text of Dune and convert it into a DataFrame.
  Data includes sentiments and top 5 words per chapter.

  Input: text (str)
  Output: duneData (DataFrame)
  '''
  if type(text)!=str:
    raise TypeError('text must be a string!')
  if text=='':
    raise ValueError('text cannot be empty!')

  # setup for unpolished data
  separated, tokenized=separateAndTokenize(text)
  chSentiments=makeChSentiments(separated)
  chTop5Words, chTop5Nums=top5Chapter(chWordCounts(tokenized))

  # This will be our polished data
  chDupes=[] # duplicates each chapter number 5 times (for each top word)
  sentDupes=[] # duplicates each sentiment 5 times (for each top word)
  ranks=[] # ranks 1 through 5 repeated 48 times (for each chapter)
  top5WordsCombined=[] # merges the lists within the chTop5Words list
  top5NumsCombined=[] # merges the lists within the chTop5Nums list

  for i in range(len(separated)):
    for j in range(5):
      chDupes.append(i+1)
      sentDupes.append(chSentiments[i])
      ranks.append(j+1)
      top5WordsCombined.append(chTop5Words[i][j])
      top5NumsCombined.append(chTop5Nums[i][j])

  duneData=pd.DataFrame({
      'Chapter': chDupes,
      'Sentiment': sentDupes,
      'Rank': ranks,
      'Word': top5WordsCombined,
      'Count': top5NumsCombined
  })

  return duneData


def makeWordVis(duneData):
  '''
  Creates a bar chart visualization of word counts in Dune.

  Input: duneData (pd.core.frame.DataFrame)
  Output: figWords
  '''
  if type(duneData)!=pd.core.frame.DataFrame:
    raise TypeError('duneData must be a DataFrame!')

  # color palette that was extremely tedious to copy over
  palette=['#FF0000', '#FF8800', '#E9FF00', '#9CFF00', '#00FF2A', '#00D897', '#00BBFF', '#002EFF', '#8300FF', '#E500FF', '#FF008C',
           '#DF0000', '#DF7600', '#CBDF00', '#88DF00', '#00DF24', '#00BC84', '#00A3DF', '#0028DF', '#7200DF', '#C800DF', '#DF007A',
           '#FF3B3B', '#FFA23B', '#EDFF3B', '#B3FF3B', '#3BFF5A', '#3BE0AF', '#3BCAFF', '#3B5EFF', '#9F3BFF', '#EA3BFF', '#FF3BA6',
           '#9F0000', '#9F5400', '#919F00', '#619F00', '#009F1A', '#00865E', '#00749F', '#001C9F', '#51009F', '#8E009F', '#9F0057',
           '#FF7373', '#FFBD73', '#F2FF73', '#C8FF73', '#73FF8A', '#73E9C5', '#73D9FF', '#738BFF', '#BA73FF', '#F073FF', '#FF73BF',
           '#740000', '#743D00', '#697400', '#477400', '#007412', '#016446', '#005474', '#001474', '#3C0074', '#680074', '#74003F',
           '#FFA0A0', '#FFD2A0', '#F6FFA0', '#DAFFA0', '#A0FFAF', '#A0F0D8', '#A0E5FF', '#A0B1FF', '#D1A0FF', '#F5A0FF', '#FFA0D4']

  figWords=px.bar(duneData, x='Chapter', y='Count', color='Word', title='Top 5 Words Per Chapter in Dune', hover_name='Word', color_discrete_sequence=palette)
  return figWords


def makeSentVis(duneData):
  '''
  Creates a bar chart visualization of sentiments per chapter of Dune

  Input: duneData (pd.core.frame.DataFrame)
  Output: figSent
  '''
  if type(duneData)!=pd.core.frame.DataFrame:
    raise TypeError('duneData must be a DataFrame!')

  sentiments=[]
  i=0

  for sent in duneData['Sentiment']: # getting rid of duplicated sentiments
    if i % 5 == 0:
      sentiments.append(sent) # once every five entries
    i+=1

  figSent=px.bar(duneData, x=list(range(1,49)), y=sentiments, color=sentiments, title='Sentiments per Chapter of Dune')
  figSent.update_layout(xaxis_title='Chapter', yaxis_title='Sentiment')
  return figSent
