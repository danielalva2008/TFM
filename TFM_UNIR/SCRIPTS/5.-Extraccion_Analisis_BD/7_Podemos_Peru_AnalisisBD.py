#!/usr/bin/env python
# coding: utf-8

# **Analisis del Partido Podemos Peru**

# In[1]:


#!pip install tweepy
#!pip install pymongo
import json
import tweepy
import pandas as pd
import re
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from datetime import datetime as dt
import numpy as np
from os import path
import pymongo
from pymongo import MongoClient
from pprint import pprint
import configparser
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
try:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import pickle
import funciones 
from funciones import (DIACRITICAL_VOWELS, SLANG, stop_words,SLANG_SP_SN,stop_words_p)
from funciones import eliminarhtml,text_to_wordlist,text_to_wordlist_wc,encoding_emoji


# In[2]:


#Descargamos los datos de Twitter
from tweepy import OAuthHandler
#Claves del acceso
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api=tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True,parser=tweepy.parsers.JSONParser())


# In[3]:


#guardamos los datos en archivo JSON
count = 3000
date_since="2019-10-01"
search_words="#PodemosPeru"
api = tweepy.API(auth)                                                                                                 
results = [status._json for status in tweepy.Cursor(api.search, q=search_words, count=count,since=date_since, tweet_mode='extended', lang='es').items()]
results
with open('PodemosPeruS1OK.json', 'w') as file:
    json.dump(results, file, indent=4)


# In[4]:


twet_data= open('PodemosPeruS1OK.json', 'r').read()
twet = json.loads(twet_data)


# In[5]:


with open('PodemosPeruS1OK.json') as train_file:
    dict_train = json.load(train_file, encoding='latin-1')
tweets = pd.DataFrame.from_dict(dict_train)
len(tweets.index)


# In[6]:


#Almacenamos los datos mas relevantes
question_list = list()
for question in tweets.source:
  question_list.append(eliminarhtml(str(question).strip()))
df = pd.DataFrame(question_list, columns =['source']) 
df['id_str']         = tweets['id_str']
df['created_at']     = tweets['created_at']
df['full_text']      = tweets['full_text']
df['favorite_count'] = tweets['favorite_count']
df['retweet_count']  = tweets['retweet_count']

df_3sub=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tweets['user'].items() ]))
df_3sub.reset_index(col_level=0)
df3_cand=df_3sub.T

df['screen_name']    = df3_cand['screen_name']
df['location']       = df3_cand['location']
df['full_text1']     = tweets['full_text']
df.drop_duplicates('full_text', keep="last", inplace=True)
df.sort_index(inplace=True) 

#Normalizamos el campo fecha
df['created_at'] = pd.to_datetime(df.created_at)
df['created_at'] = df['created_at'].dt.normalize()
df['numero']=1
df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d')
df = df.reset_index(drop=True)


# In[7]:


question_list = list()
for question in df.full_text:
  question_list.append(text_to_wordlist(str(question).strip()))
df1 = pd.DataFrame(question_list, columns =['full_text'])


# In[8]:


with open('tokenizador/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[9]:


Dcnn2 = load_model("dcnn")
pred = df1.full_text.apply(tokenizer.encode)
pred = tf.keras.preprocessing.sequence.pad_sequences(pred,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=40)
len(np.argmax(Dcnn2.predict(pred),axis=1))


# In[10]:


# Analisis de Sentimiento
my_array=np.argmax(Dcnn2.predict(pred),axis=1)
dfp = pd.DataFrame(my_array, columns = ['POS'])
dfp['POS1'] = dfp['POS'].map({0:"Negativo", 1:"Neutro",2:"Positivo"})
df['POS']=dfp['POS1']
df['POS1']=dfp['POS1'].map({"Negativo":"cant_tot_negativos", "Neutro":"cant_tot_neutro","Positivo":"cant_tot_positivos"})
total_podemos_peru = df.groupby(['POS1']).agg(
                                  {'numero': 'sum'
                                  }).reset_index()


# In[11]:


df_analisis_sentimiento_podemos_peru = total_podemos_peru.T
df_analisis_sentimiento_podemos_peru.reset_index(drop=True,inplace=True)
df_analisis_sentimiento_podemos_peru = df_analisis_sentimiento_podemos_peru.rename(columns={0:df_analisis_sentimiento_podemos_peru.loc[0, 0]})
df_analisis_sentimiento_podemos_peru = df_analisis_sentimiento_podemos_peru.rename(columns={1:df_analisis_sentimiento_podemos_peru.loc[0, 1]})
df_analisis_sentimiento_podemos_peru = df_analisis_sentimiento_podemos_peru.rename(columns={2:df_analisis_sentimiento_podemos_peru.loc[0, 2]})
df_analisis_sentimiento_podemos_peru['cant_tot_extraidos']=len(df)
df_analisis_sentimiento_podemos_peru.drop([0],axis=0,inplace=True)
#df_analisis_sentimiento_podemos_peru


# In[12]:


lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = stopwords.words('spanish')


# In[13]:


question_list = list()
for question in df.full_text1:
  question_list.append(text_to_wordlist_wc(str(question).strip()))
df1 = pd.DataFrame(question_list, columns =['full_text1']) 
#df1


# In[14]:


comment_words = '' 
for val in df1.full_text1: 
    val = str(val) 
    # split the value 
    tokens = val.split() 
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += " ".join(tokens)+" "


# In[15]:


tokenized_word=word_tokenize(comment_words)
#print(tokenized_word)


# In[16]:


fdist = FreqDist(tokenized_word)
print(fdist)
xd=fdist.most_common(20)


# In[17]:


# Analisis de tendencia app
df_analisis_tendencia_podemos_peru = pd.DataFrame(data=xd,columns=['termino','frecuencia'])
source_df = df.groupby(('source')).numero.sum()
source_sorted = sorted(source_df.items(), key=lambda x: x[1], reverse=True)
source_t= pd.DataFrame(data=source_sorted,columns=['source','cantidad'])
#df_analisis_tendencia_podemos_peru


# In[18]:


df_t_usuarios_podemos_peru = df.groupby(['created_at','POS']).agg(
                                  {'numero': 'sum'
                                  }).reset_index()

df_t_usuarios_podemos_peru.rename(columns={'created_at': 'fecha', 
                                    'POS': 'sentimiento','numero':'tweet'}, inplace=True)
#df_t_usuarios_podemos_peru


# In[19]:


#conectamos a la base de datos
mongod_connect = 'mongodb://localhost:27017'
client = MongoClient(mongod_connect)
db =client['db_twitter']


# In[20]:


#creamos la tabla analisis_sentimiento_podemos_peru
analisis_sentimiento_podemos_peru = db.analisis_sentimiento_podemos_peru 
records = json.loads(df_analisis_sentimiento_podemos_peru.T.to_json()).values()
db.analisis_sentimiento_podemos_peru.insert_many(records)


# In[21]:


df_t_podemos_peru =pd.DataFrame()
df_t_podemos_peru['id_str']         =df['id_str']
df_t_podemos_peru['screen_name']    =df['screen_name']
df_t_podemos_peru['created_at']     =df['created_at']
df_t_podemos_peru['full_text']      =df['full_text']
df_t_podemos_peru['favorite_count'] =df['favorite_count']
df_t_podemos_peru['retweet_count']  =df['retweet_count']
df_t_podemos_peru['location']       =df['location']
df_t_podemos_peru['source']         =df['source']
df_t_podemos_peru['pos']            =df['POS']


# In[22]:


df_usuario_podemos_peru = db.t_usuario_podemos_peru
records = json.loads(df_t_podemos_peru.T.to_json()).values()
db.t_podemos_peru.insert_many(records)


# In[23]:


analisis_tendencia_podemos_peru = db.analisis_tendencia_podemos_peru
records = json.loads(df_analisis_tendencia_podemos_peru.T.to_json()).values()
db.analisis_tendencia_podemos_peru.insert_many(records)


# In[24]:


t_usuarios_podemos_peru = db.t_usuarios_podemos_peru
records = json.loads(df_t_usuarios_podemos_peru.T.to_json()).values()
db.t_usuarios_podemos_peru.insert_many(records)


# In[ ]:





# In[ ]:




