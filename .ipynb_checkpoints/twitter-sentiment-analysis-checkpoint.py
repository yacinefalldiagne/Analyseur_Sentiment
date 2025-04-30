#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os


# In[ ]:


sns.set_palette('pastel')

sns.set_style('darkgrid')


# # Reading data

# Our objective is to calssify sentiments of various tweet

# In[ ]:


train_df=pd.read_csv('train.csv',encoding='ISO-8859-1')

train_df


# In[ ]:


test_df=pd.read_csv('test.csv',encoding='ISO-8859-1')

test_df


# In[ ]:


print(f'Shape of training data:{train_df.shape}')

print(f'Shape of test data:{test_df.shape}')


# Training data has an extra 'selected text'column to capture the sentiment.
# 
# 
# 

# In[ ]:


#print(train_df.columns.tolist())
train_df.info()


# In[ ]:


#Removing the null column

train_df=train_df.dropna(how='any',axis=0)


# In[ ]:


train_df.shape


# In[ ]:


sns.countplot(train_df,x='sentiment')


# We can see that the dataframe is pretty much balanced

# In Text processing, we need to represent the words as numeric vectors.
# 
# 
# 
# But first of all, we have to preprocess the input

# # **Preprocessing Input**

# In[ ]:


train_df['text'].head(5)


# Tweets are mostly casual mode which may also contain some phone nos,URLs and Hashtags.
# 
# We have to choose the tokenizer which best deals with these
# 
# 
# 

# In[ ]:


#Analyzing texts

for i in np.arange(1,20):

  print(train_df['text'][i])


# In[ ]:


import string

import re


# **Websites** beginning with https/www or **email** rarely convey meaning information, we will remove this with  regular expression

# In[ ]:


pattern_web =  r"(([\w]+:)?//)?(([\d\w]|%[a-fA-F\d]{2,2})+(:([\d\w]|%[a-fA-f\d]{2,2})+)?@)?([\d\w][-\d\w]{0,253}[\d\w]\.)+[\w]{2,4}(:[\d]+)?(/([-+_~.\d\w]|%[a-fA-f\d]{2,2})*)*(\?(&?([-+_~.\d\w]|%[a-fA-f\d]{2,2})=?)*)?(#([-+_~.\d\w]|%[a-fA-f\d]{2,2})*)?"


# In[ ]:


str='''Im going to http://www.example.com google.com https://example.co.uk'''

re.sub(pattern_web,'',str)


# In[ ]:


pattern_email=r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*)@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"


# In[ ]:


str='''Send mail to user@example.com person-1992@fg.co.in manu@epl.co.uk'''

re.sub(pattern_email,' ',str)


# Tweets contain **hashtags** and **handles**
# 
# From hashtags, # should be removed leaving the base word
# 
# Handles should be removed as they are used to refer to someone

# In[ ]:


pattern_hash=r'#(\w+)'


# In[ ]:


str=''' I'm cool  #Summer2024 but my phone# # 444 Feat#UK '''

re.sub(pattern_hash," \\1",str)  #Keeps the part of expression removing hashtags only


# In[ ]:


pattern_handle=r'@\w+'  # Only matched if @ followed by alphanumeric characters & underscore


# In[ ]:


str=''' @Mike Is ur email gg@hj.com '''

re.sub(pattern_handle," ",str)


# Twitter also contains **emojis** which can convey important informations, we will use emoji package to convert those into expressions

# In[ ]:


get_ipython().system('pip install emoji')


# In[ ]:


import emoji


# In[ ]:


str='I love python 😊 '

emoji.demojize(str)


# Many tweets use **contractions** of full words and these must be expanded to reveal their meaning

# In[ ]:


get_ipython().system('pip install contractions')


# In[ ]:


import contractions


# In[ ]:


contractions.safety_keys


# In[ ]:


str="I'd have responded, if I were going"

contractions.fix(str)


# Often in tweets, some **characters are repeated** no of times to accentuate their meaning, for eg way to waaaay etc.
# 
# For this , we will contract the repeated characters

# In[ ]:


pattern_repeat=r'([A-Za-z])\1{2,}'  # Usage of #1 implies that 1st character used has to match.{2,}-repeat more than 2 times



# In[ ]:


str=''' U r so greaat,lets be friendssss cool '''

re.sub(pattern_repeat,r'\1',str)


# Now that we have partially pre-processed tweets, there are still some processing that needs to be done:
# 
# 1.**Numbers**: They reveal important information,but since we are basically classifying sentiment , their impact will be minimum.**Numbers will be removed**
# 
# 
# 
# 2.**Proper Nouns & lower casing**: It will be very helpful if we can separate the proper nouns from Uppercase letters , but in tweets Upper case apperas all over the place due to casual nature.Morever, in sentiment analysis, proper nouns dont reveal much info**All letters to be lowercased**
# 
# 
# 
# 3.**Punctuation**: Mostly used ungramatically, and since we have already taken care of contractions,handles and hashtags,their retention will be confusing.**Remaining punctuations should be removed**
# 

# In[ ]:


string.punctuation


# In[ ]:


pattern_num=r'[0-9]'


# In[ ]:


re.sub(pattern_num," ","i've 4 $ %")


# In[ ]:


#Defing a function to apply all pre-processing steps



def pre_process(tweet):

  t_mod=re.sub(pattern_web,'',tweet)        #Removing URLs

  t_mod=re.sub(pattern_email,' ',t_mod)     #Removing email

  t_mod=re.sub(pattern_hash," \\1",t_mod)   #Removing '#'

  t_mod=re.sub(pattern_handle," ",t_mod)    #Removing handles

  t_mod=emoji.demojize(t_mod)               #De-emojifying


  pattern_apos=r'`'                         #Replacing ` with ' to be used in contraction

  t_mod=re.sub(pattern_apos,"'",t_mod)


  t_mod=contractions.fix(t_mod)              #Removing contractions


  t_mod=re.sub(pattern_repeat,r'\1',t_mod)   #Normalising repeated characters


  pattern_num=r'[0-9]'

  t_mod=re.sub(pattern_num," ",t_mod)       #Removing numbers


  pattern_punc="[" + re.escape(string.punctuation) + "]"       #Removing punctuation

  t_mod=re.sub(pattern_punc," ",t_mod)


  t_mod=t_mod.lower()                       #lowering characters


  pattern_space=r'\s+'                      #Removing extra white space chars

  t_mod=re.sub(pattern_space," ",t_mod)


  return t_mod


# In[ ]:


train_df['cleaned_text']=train_df['text'].apply(pre_process)


# In[ ]:


train_df


# # Tokenizing

# In[ ]:


from nltk.tokenize import TreebankWordTokenizer


# In[ ]:


tokenizer=TreebankWordTokenizer()


# In[ ]:


train_df['tokens']=train_df['cleaned_text'].apply(lambda x:tokenizer.tokenize(x))


# In[ ]:


train_df


# In[ ]:


#Checking the length  of tokens

train_df['token_len']=train_df['tokens'].apply(lambda x:len(x))





sns.kdeplot(train_df,x='token_len',legend='All',fill=True,label='All')

sns.kdeplot(train_df[train_df['sentiment']=='neutral'],x='token_len',label='neutral',fill=True)

sns.kdeplot(train_df[train_df['sentiment']=='positive'],x='token_len',label='positive',fill=True)

sns.kdeplot(train_df[train_df['sentiment']=='negative'],x='token_len',label='negative',fill=True)

plt.legend()

plt.show()


# We can see that there is no notable difference in distribution of token length among sentiments, though'Neutral sentiment' seems to be of smaller length

# In[ ]:


from collections import Counter


# In[ ]:


ct=Counter()

for token_list in train_df['tokens']:

  ct.update(token_list)


# In[ ]:


sum(list(ct.values()))


# In[ ]:


#Most common words in tweets

dict(ct.most_common(50))


# In[ ]:


#Distribution of 50 most common tokens

plt.figure(figsize=(10,5))

plt.bar(x=dict(ct.most_common(50)).keys(),height=dict(ct.most_common(50)).values())

plt.xticks(rotation=45)

plt.xlabel('most common words in tweets')

plt.show()


# We can see that the most common words are the general words, which though important in context, will be useful for differentiating between sentiments
# 
# 
# 
# We will **filter out all the common words in english dict**

# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


import nltk

nltk.download('stopwords')


# In[ ]:


stopword_list=stopwords.words('english')


# In[ ]:


train_df['tokens_wo_stopwords']=train_df['tokens'].apply(lambda x:[i for i in x if i not in stopword_list])


# In[ ]:


train_df


# Using **lemmatization** for identifying the root words

# In[ ]:


from nltk.stem import WordNetLemmatizer


# In[ ]:


nltk.download('wordnet')


# In[ ]:


get_ipython().system('unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/')


# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


lemmatizer.lemmatize('gives')


# In[ ]:


train_df['tokens_lemma']=train_df['tokens_wo_stopwords'].apply(lambda x:[lemmatizer.lemmatize(i) for i in x])

train_df['tokens_lemma']


# In[ ]:


#Counters for listing tokens



ct_all=Counter()

ct_neutral=Counter()

ct_positive=Counter()

ct_negative=Counter()



for token_list in train_df['tokens_lemma']:

  ct_all.update(token_list)



for token_list in train_df[train_df['sentiment']=='neutral']['tokens_lemma']:

  ct_neutral.update(token_list)



for token_list in train_df[train_df['sentiment']=='positive']['tokens_lemma']:

  ct_positive.update(token_list)



for token_list in train_df[train_df['sentiment']=='negative']['tokens_lemma']:

  ct_negative.update(token_list)


# In[ ]:


dict(ct_all.most_common(50)).values()


# In[ ]:


#Distribution of 50 most common tokens

plt.figure(figsize=(10,5))

plt.bar(x=dict(ct_all.most_common(50)).keys(),height=dict(ct_all.most_common(50)).values())

plt.xticks(rotation=45)

plt.xlabel('most common words in tweets')

plt.tight_layout()

plt.show()


# In[ ]:


#Distribution of 50 most common positive tokens

plt.figure(figsize=(10,5))

plt.bar(x=dict(ct_positive.most_common(50)).keys(),height=dict(ct_positive.most_common(50)).values())

plt.xticks(rotation=45)

plt.xlabel('most common words in positive tweets')

plt.tight_layout()

plt.show()


# In[ ]:


#Distribution of 50 most common negative tokens

plt.figure(figsize=(10,5))

plt.bar(x=dict(ct_negative.most_common(50)).keys(),height=dict(ct_negative.most_common(50)).values())

plt.xticks(rotation=45)

plt.xlabel('most common words in negative tweets')

plt.tight_layout()

plt.show()


# In[ ]:


get_ipython().system('pip install wordcloud')


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


def generate_wordcloud(list_,label_list):

  plt.figure(figsize=(10,20))

  k=1

  for i in list_:

    plt.subplot(len(list_),1,k)

    word_cloud=WordCloud()

    word_cloud.generate_from_frequencies(i)

    plt.imshow(word_cloud)

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.xlabel(label_list[k-1])

    k+=1


# In[ ]:


generate_wordcloud([ct_all,ct_positive,ct_negative,ct_neutral],['All','Positive','Negative','Neutral'])


# Now that we have visualized the tokens , we need to convert our tokens into numeric vectors.
# 
# 
# 
# There are several ways to convert but we will use word embeddings to get the most feature-rich representation of our tokens
# 
# 
# 
# Further we will use the original token list without removing stopwords or lemmatozation,for maximum representation of our data

# In[ ]:


#Getting the final data

train_df_chosen=train_df[['tokens','sentiment']]

train_df_chosen


# In[ ]:


#Converting target class to on-hot vector

train_df_final=pd.get_dummies(train_df_chosen,columns=['sentiment'],dtype='int')



train_df_final['sentence']=train_df_final['tokens'].apply(lambda x:" ".join(x))   #Converting tokens back to string to process in Vectorizer

train_df_final


# In[ ]:


#Checking the length of sentence
train_df_final['sentence_length']=train_df_final['sentence'].apply(lambda x:len(x.split()))
train_df_final['sentence_length'].value_counts().sort_index().plot.bar()


# In[ ]:


#Dropping the occurences of null length
train_df_final=train_df_final.drop(train_df_final[train_df_final['sentence_length']==0].index,axis=0)


# In[ ]:


#Getting the test dataset

test_df


# In[ ]:


test_df.info()


# In[ ]:


#Dropping the null rows
test_df=test_df.dropna(how='any',axis=0)


# In[ ]:


#Pre-processing and tokenizing

test_df['cleaned_text']=test_df['text'].apply(pre_process)        #Applying pre-processing


test_df['tokens']=test_df['cleaned_text'].apply(lambda x:tokenizer.tokenize(x))   #Applying tokenizer


# In[ ]:


test_df_chosen=test_df[['tokens','sentiment']]

test_df_final=pd.get_dummies(test_df_chosen,columns=['sentiment'],dtype='int')



test_df_final['sentence']=test_df_final['tokens'].apply(lambda x:" ".join(x))

test_df_final


# In[ ]:


#Checking the length of sentence
test_df_final['sentence_length']=test_df_final['sentence'].apply(lambda x:len(x.split()))
test_df_final['sentence_length'].value_counts().sort_index().plot.bar()


# In[ ]:


#Dropping the occurences of null length
test_df_final=test_df_final.drop(test_df_final[test_df_final['sentence_length']==0].index,axis=0)


# In[ ]:


train_df_features=train_df_final['sentence']

train_df_targets=train_df_final[['sentiment_negative','sentiment_neutral','sentiment_positive']]



test_df_features=test_df_final['sentence']

test_df_targets=test_df_final[['sentiment_negative','sentiment_neutral','sentiment_positive']]


# In[ ]:


#Splitting into train and validation sets

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val=train_test_split(train_df_features,train_df_targets,random_state=42,shuffle=True,test_size=0.2)


# In[ ]:


X_test=test_df_features

y_test=test_df_targets


# In[ ]:


#Checking shape of our dataset

print(f"Shape of X train:{X_train.shape}")

print(f"Shape of X val:{X_val.shape}")

print(f"Shape of X test:{X_test.shape}")

print(f"Shape of Y train:{y_train.shape}")

print(f"Shape of Y val:{y_val.shape}")

print(f"Shape of Y test:{y_test.shape}")


# # Vectorizing

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization


# In[ ]:


#Running  a counter to list all tokens

ct=Counter()

for token_list in train_df_final['tokens']:

  ct.update(token_list)


# In[ ]:


#Getting the max token list
print(f'No of tokens present in our vocabulary:{len(ct)}')

max_tokens=len(ct)


# In[ ]:


token_list=list(sorted(ct.keys()))


# Outsput seq length is kept as 35, bcz it is the max length of cleaned tweets

# In[ ]:


text_vectorization = TextVectorization(max_tokens,output_mode="int",output_sequence_length=35,pad_to_max_tokens=True) #Limiting sequence length to 30


# In[ ]:


#Adapting the vocabulary

text_vectorization.adapt(token_list)


# In[ ]:


text_vectorization.vocabulary_size()


# In[ ]:


#Vectorizing the tweets
X_train_vector=text_vectorization(X_train)
X_val_vector=text_vectorization(X_val)
X_test_vector=text_vectorization(X_test)

X_train_vector


# In[ ]:


#Defining a function to change 0s for masking purpose

def change_padding_values(tensor,mask_value):

    mask = tf.equal(tensor, 0)
    tensor_new = tf.where(mask, mask_value, tensor)

    return tensor_new


# In[ ]:


X_train_vector=change_padding_values(X_train_vector,mask_value=-999)
X_val_vector=change_padding_values(X_val_vector,mask_value=-999)
X_test_vector=change_padding_values(X_test_vector,mask_value=-999)


# # Using GloVe embeddings

# Now that we have our dataset ready, we will apply gloVe vectorization for the tokens

# In[ ]:


get_ipython().system('pip install wget unzip')


# In[ ]:


import wget
url = 'http://nlp.stanford.edu/data/glove.6B.zip'
wget.download(url)
# !unzip -q glove.6B.zip


# In[ ]:


#Using glove 100-D model

with open('glove.6B.100d.txt','r') as f:

  print(f.readline())


# We can see structure of the file, we will parse separately the key and the values

# In[ ]:


embeddings_index = {}

with open('glove.6B.100d.txt') as f:



  for line in f:

    word, coefs = line.split(maxsplit=1)

    coefs = np.fromstring(coefs, "f", sep=" ")

    embeddings_index[word] = coefs


# In[ ]:


feature_dim=100


# In[ ]:


print(f'Length of word index dict:{len(embeddings_index)}')


# Now creating an Embedding Matrix based on the words present in our tokens

# In[ ]:


embedding_matrix=np.zeros((max_tokens,feature_dim))


# In[ ]:


for i,word in enumerate(text_vectorization.get_vocabulary()):

  embedding_vector=embeddings_index.get(word)

  if embedding_vector is not None:

    embedding_matrix[i,:]=embedding_vector


print(f'Shape of embedding matrix={embedding_matrix.shape}')


# In[ ]:


null_arrays=np.sum((np.any(embedding_matrix,axis=1)==0))

null_arrays


# We can see that there are 4656 null arrays in our matrices which means that 4656 of our words did not have corresponding glove vectors

# ### Training the model using gloVe embeddings

# In[ ]:


from keras.layers import Embedding,Dense,Dropout,Input,BatchNormalization,LSTM,Bidirectional,Masking

from keras.models import Model


# For the embedding layer, trainable to be kept False to fix the weights and zero mask to be enabled to skip the zero occurence while using Bideirectional RNN

# In[ ]:


#Defining Embedding layer

embedding=Embedding(input_dim=max_tokens,output_dim=feature_dim,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,mask_zero=False)


# Using Bidirectional LSTM for model

# In[ ]:


#Defining model with gloVe embeddings

def model_glove(max_seq_length,max_tokens,feature_dim,embedding_matrix,mask_value):
    
    #Defining embedding layer
    embedding=Embedding(input_dim=max_tokens,output_dim=feature_dim,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,mask_zero=False)
    
    #Defining the model
    inputs=Input(shape=(max_seq_length,))
    
    embedded=embedding(inputs)
    
    masked_layer=Masking(mask_value=mask_value)(embedded)
    
    
    x=Bidirectional(LSTM(1024,return_sequences=True,recurrent_dropout=0.2))(masked_layer)

    x=Bidirectional(LSTM(1024,recurrent_dropout=0.2))(x)
    
    outputs=Dense(3,activation='softmax')(x)
    
    
    
    model=Model(inputs,outputs)
    
    

    model.compile(optimizer='adamax',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


# In[ ]:


callback=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)


# In[ ]:


model=model_glove(max_seq_length=35,max_tokens=22520,feature_dim=100,embedding_matrix=embedding_matrix,mask_value=-999)
model.summary()


# In[ ]:


model.fit(X_train_vector,np.array(y_train), validation_data=[X_val_vector,np.array(y_val)], epochs=30,batch_size=128,callbacks=[callback])


# In[ ]:


model.evaluate(X_test_vector,np.array(y_test))


# In[ ]:


y_test_prob=model.predict(X_test_vector)

y_test_pred=np.argmax(y_test_prob,axis=1)

y_test_pred


# In[ ]:


y_test_org=np.argmax(y_test,axis=1)


# In[ ]:


from sklearn import metrics


# In[ ]:


df_metrics=pd.DataFrame({'glove embeddings':[metrics.accuracy_score(y_test_org,y_test_pred),
              metrics.recall_score(y_test_org,y_test_pred,average="weighted"),
              metrics.precision_score(y_test_org,y_test_pred,average="weighted"),
              metrics.f1_score(y_test_org,y_test_pred,average="weighted")]},
              index=[['Accuracy','Recall','Precision','F1-Score']])

df_metrics


# In[ ]:


conf_matrix=metrics.confusion_matrix(y_test_org,y_test_pred)

sns.heatmap(conf_matrix,cmap='viridis',annot=True,fmt='.0f',xticklabels=test_df_targets.columns, yticklabels=test_df_targets.columns)

plt.xlabel('Actual')
plt.ylabel('Predicted')


plt.show()


# Finally, we can see that we have received an accuracy of around 0.7 on Test Data,
# 
# but most of the mis-classifications has been recored for neutral tweets or have been misclassified as neutral
# 
# 
# 
# There are very few positive and negative tweets which have been misclassified as the opposite

# # Using ELMo embeddings

# We will try to improve upon the previous model by using ELMo embeddings, which can vectorize based on taking words meaning and position in context

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub


# In[ ]:


#Loading the elmo model

elmo = hub.load("https://tfhub.dev/google/elmo/3")


# In[ ]:


#Checking the vectorization of 1st tweet

embeddings_elmo = elmo.signatures["default"](tf.constant([X_train.values.tolist()[0]]))["elmo"]

embeddings_elmo[0]


# In[ ]:


#Defining a function to create elmo embedding matrix

def create_elmo_matrix(sentence_array,elmo,max_length,feature_dim,mask_value):
    len_array=len(sentence_array.values.tolist())
    embedding_matrix=np.zeros((len_array,max_length,feature_dim))  #Initializing the matrix


    for i,k in enumerate(sentence_array.values.tolist()):
    
        embeddings= elmo.signatures["default"](tf.constant([k]))["elmo"]
        
            
        if embeddings[0].shape[0]<max_length:
            embedding_matrix[i,:embeddings[0].shape[0],:]=embeddings[0]  #Filling matrix with embedding vector
            embedding_matrix[i,embeddings[0].shape[0]:,:]=mask_value     #Filling unfilled matrix with masked value
            
        else :
            embedding_matrix[i,:max_length,:]=embeddings[0,:max_length]  #Filling matrix with embedding vector

    return embedding_matrix 


# In[ ]:


#Creating elmo vectors
X_train_elmo=create_elmo_matrix(X_train,elmo,max_length=35,feature_dim=1024,mask_value=-999)
X_val_elmo=create_elmo_matrix(X_val,elmo,max_length=35,feature_dim=1024,mask_value=-999)
X_test_elmo=create_elmo_matrix(X_test,elmo,max_length=35,feature_dim=1024,mask_value=-999)


# In[ ]:


#Defining function for elmo model
def model_elmo(max_seq_length,feature_dim,mask_value):

    inputs=Input(shape=(max_seq_length,feature_dim))

    
    masked_layer=Masking(mask_value=mask_value)(inputs)
    
    
    x=Bidirectional(LSTM(2048,return_sequences=True,recurrent_dropout=0.2))(masked_layer)
    
    x=Bidirectional(LSTM(2048,recurrent_dropout=0.2))(x)
    
    outputs=Dense(3,activation='softmax')(x)
    
    
    
    model=Model(inputs,outputs)

    model.compile(optimizer='adamax',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


# In[ ]:


callback=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)


# In[ ]:


model=model_elmo(max_seq_length=35,feature_dim=1024,mask_value=-999)
model.summary()


# In[ ]:


model.fit(X_train_elmo,np.array(y_train), validation_data=[X_val_elmo,np.array(y_val)], epochs=30,batch_size=128,callbacks=[callback])


# In[ ]:


model.evaluate(X_test_elmo,np.array(y_test))


# In[ ]:


y_test_prob=model.predict(X_test_elmo)

y_test_pred=np.argmax(y_test_prob,axis=1)

y_test_pred


# In[ ]:


y_test_org=np.argmax(y_test,axis=1)


# In[ ]:


df_metrics['elmo embeddings']=[metrics.accuracy_score(y_test_org,y_test_pred),
              metrics.recall_score(y_test_org,y_test_pred,average="weighted"),
              metrics.precision_score(y_test_org,y_test_pred,average="weighted"),
              metrics.f1_score(y_test_org,y_test_pred,average="weighted")]
df_metrics


# Using ELMo embeddings improves our accuracy by 3 percent points

# # Using Transformers

# We will use Huggingface's DistilBERT Transformer 

# In[ ]:


import tensorflow as tf
from tensorflow import keras



# In[ ]:


from huggingface_hub import login


# In[ ]:


#Importing AutoTekenizer

from transformers import AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# In[ ]:


#Defining a function to encode
def batch_encode(tokenizer, texts, batch_size=256, max_length=128):
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             pad_to_max_length =True,
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])


    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


# In[ ]:


#Encoding 
X_train_ids, X_train_attention = batch_encode(auto_tokenizer, X_train.tolist())
X_val_ids, X_val_attention = batch_encode(auto_tokenizer, X_val.tolist())
X_test_ids, X_test_attention = batch_encode(auto_tokenizer, X_test.tolist())


# 

# In[ ]:


#Loading the model
from transformers import TFDistilBertForSequenceClassification,DistilBertConfig



model_distilBERT = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=3)

model_distilBERT.summary()


# In[ ]:


#Updating dropout

model_distilBERT.config.attention_dropout=0.2
model_distilBERT.config.dropout=0.2


# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model_distilBERT.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# In[ ]:


callback=tf.keras.callbacks.EarlyStopping(patience=5,monitor='val_loss',restore_best_weights=True)


# In[ ]:


model_distilBERT.fit([X_train_ids, X_train_attention],np.array(y_train), validation_data=[[X_val_ids, X_val_attention],np.array(y_val)], epochs=20,batch_size=128,callbacks=[callback])


# In[ ]:


model_distilBERT.evaluate([X_test_ids, X_test_attention],np.array(y_test))


# In[ ]:


y_test_prob=model_distilBERT.predict([X_test_ids, X_test_attention])

y_test_pred=np.argmax(y_test_prob.logits,axis=1)


# In[ ]:


y_test_org=np.argmax(y_test,axis=1)


# In[ ]:


from sklearn import metrics
df_metrics['distilbert']=[metrics.accuracy_score(y_test_org,y_test_pred),
              metrics.recall_score(y_test_org,y_test_pred,average="weighted"),
              metrics.precision_score(y_test_org,y_test_pred,average="weighted"),
              metrics.f1_score(y_test_org,y_test_pred,average="weighted")]
df_metrics.round(2)


# In[ ]:


conf_matrix=metrics.confusion_matrix(y_test_org,y_test_pred)

sns.heatmap(conf_matrix,cmap='viridis',annot=True,fmt='.0f',xticklabels=test_df_targets.columns, yticklabels=test_df_targets.columns)

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('DistilBERT Classifier')


plt.show()


# **As distilBERT model performs the best, same model will be saved and used for classification of tweets**

# In[ ]:


#Making a directory and saving the model

if not os.path.exists('tweet_sentiment_analysis'):
     os.makedirs('tweet_sentiment_analysis')

model_distilBERT.save(os.path.join('tweet_sentiment_analysis','model_distilBERT'),save_format="tf")


# # Compiling all functions together

# In[ ]:


import re
import contractions
import emoji

def pre_process(tweet):

  t_mod=re.sub(pattern_web,'',tweet)        #Removing URLs

  t_mod=re.sub(pattern_email,' ',t_mod)     #Removing email

  t_mod=re.sub(pattern_hash," \\1",t_mod)   #Removing '#'

  t_mod=re.sub(pattern_handle," ",t_mod)    #Removing handles

  t_mod=emoji.demojize(t_mod)               #De-emojifying


  pattern_apos=r'`'                         #Replacing ` with ' to be used in contraction
  t_mod=re.sub(pattern_apos,"'",t_mod)


  t_mod=contractions.fix(t_mod)              #Removing contractions


  t_mod=re.sub(pattern_repeat,r'\1',t_mod)   #Normalising repeated characters


  pattern_num=r'[0-9]'

  t_mod=re.sub(pattern_num," ",t_mod)       #Removing numbers


  pattern_punc="[" + re.escape(string.punctuation) + "]"       #Removing punctuation

  t_mod=re.sub(pattern_punc," ",t_mod)


  t_mod=t_mod.lower()                       #lowering characters

  pattern_space=r'\s+'                      #Removing extra white space chars

  t_mod=re.sub(pattern_space," ",t_mod)


  return t_mod


# In[ ]:


from transformers import AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# In[ ]:


#Loading the model
model_db=tf.keras.models.load_model(os.path.join('tweet_sentiment_analysis','model_distilBERT'))
model_db.summary()


# In[ ]:


#Creating a dictionary to output sentiments
dict_sentiments=dict(enumerate(test_df_targets.columns))

dict_sentiments


# In[ ]:


#Defining a function to analyze tweet
def analyze_tweet(text,tokenizer=auto_tokenizer,model=model_db,dict_sentiments=dict_sentiments):
    cleaned_text=pre_process(text)
    tokens = auto_tokenizer(cleaned_text, return_tensors="tf",max_length=128, truncation=True, padding="max_length",
                            return_attention_mask=True, return_token_type_ids=False)
    token_input_ids=tokens['input_ids']
    token_attention_mask=tokens['attention_mask']
    
    y_score=model.predict(dict(tokens))

    y_pred=np.argmax(y_score['logits'],axis=1)

    
    return dict_sentiments[y_pred[0]]
    


# In[ ]:


#Executing the function
sentiment=analyze_tweet(test_df['text'][1],tokenizer=AutoTokenizer,model=model_db,dict_sentiments=dict_sentiments)

print(f"Text:{test_df['text'][1]} \nSentiment:{sentiment}")


# In[ ]:


#Checking on some test tweet


example_df=test_df[['text','sentiment']][0:10]
example_df['predicted']=example_df['text'].apply(analyze_tweet,tokenizer=AutoTokenizer,model=model_db,dict_sentiments=dict_sentiments)

example_df


# # Summary

# In[ ]:


df_metrics.round(3)


# Our objective was to analyze the sentiment of tweets.
# 
# Since tweets often involve informal language as well as combination of different objects like emojis, phone nos, URLs etc, it is very important to pre-process the tweets and filter out the main words which can be vectorized.
# 
# Once we have processed the tweets into clean text, we used 3 different vectorizations which achieved **accuracy** as follows:
# * Using gloVe embeddings and trained upon Bidirectional LSTM- 72%
# * Using ELMo embeddings and  trained upon Bidirectional LSTM- 75%
# * Using DistilBERT Text Classification based on Transformer from hugginface-       78%
# 
# Hence, we can conclude that  using Transformer-encoder architecture provides the best accuracy 

# In[ ]:




