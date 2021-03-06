#!/usr/bin/env python
# coding: utf-8

# **Analisis del Partido Frente Amplio**

# In[31]:


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


# In[32]:


#Descargamos los datos de Twitter
from tweepy import OAuthHandler
#Claves del acceso
consumer_key = 'eMcnfrsrAgghHJCCBdP8QStYj'
consumer_secret = 'M2FwH7ARB8MK6hq0n8HTrVErWanAsTST0nkmw11VfvuMjn8fel'
access_token = '587863980-j6aJNJaXotnCuYF5rBZ8myUYF8kCet61E5sgWWHC'
access_token_secret = 'TbSQ6OVufSk0LH6iiRpSAJBwffggEOR2Ah1ToXRQghRIV'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api=tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True,parser=tweepy.parsers.JSONParser())


# In[33]:


#guardamos los datos en archivo JSON
count = 3000
date_since="2019-10-01"
search_words="#FrenteAmplio"
api = tweepy.API(auth)                                                                                                 
results = [status._json for status in tweepy.Cursor(api.search, q=search_words, count=count,since=date_since, tweet_mode='extended', lang='es').items()]
results
with open('FrenteAmplioS1.json', 'w') as file:
    json.dump(results, file, indent=4)


# In[34]:


twet_data= open('FrenteAmplioS1.json', 'r').read()
twet = json.loads(twet_data)


# In[35]:


with open('FrenteAmplioS1.json') as train_file:
    dict_train = json.load(train_file, encoding='latin-1')
tweets = pd.DataFrame.from_dict(dict_train)
len(tweets.index)


# In[36]:


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

#dfx = dfx.reset_index(drop=True)
#Normalizamos el campo fecha
df['created_at'] = pd.to_datetime(df.created_at)
df['created_at'] = df['created_at'].dt.normalize()
df['numero']=1
df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d')
df = df.reset_index(drop=True)


# In[37]:


question_list = list()
for question in df.full_text:
  question_list.append(text_to_wordlist(str(question).strip()))
df1 = pd.DataFrame(question_list, columns =['full_text'])


# In[38]:


with open('tokenizador/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[39]:


Dcnn2 = load_model("dcnn")
pred = df1.full_text.apply(tokenizer.encode)
pred = tf.keras.preprocessing.sequence.pad_sequences(pred,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=40)
len(np.argmax(Dcnn2.predict(pred),axis=1))


# In[40]:


# Analisis de Sentimiento
my_array=np.argmax(Dcnn2.predict(pred),axis=1)
dfp = pd.DataFrame(my_array, columns = ['POS'])
dfp['POS1'] = dfp['POS'].map({0:"Negativo", 1:"Neutro",2:"Positivo"})
df['POS']=dfp['POS1']
df['POS1']=dfp['POS1'].map({"Negativo":"cant_tot_negativos", "Neutro":"cant_tot_neutro","Positivo":"cant_tot_positivos"})
total_frente_amplio = df.groupby(['POS1']).agg(
                                  {'numero': 'sum'
                                  }).reset_index()


# In[41]:


df_analisis_sentimiento_frente_amplio = total_frente_amplio.T
df_analisis_sentimiento_frente_amplio.reset_index(drop=True,inplace=True)
df_analisis_sentimiento_frente_amplio = df_analisis_sentimiento_frente_amplio.rename(columns={0:df_analisis_sentimiento_frente_amplio.loc[0, 0]})
df_analisis_sentimiento_frente_amplio = df_analisis_sentimiento_frente_amplio.rename(columns={1:df_analisis_sentimiento_frente_amplio.loc[0, 1]})
df_analisis_sentimiento_frente_amplio = df_analisis_sentimiento_frente_amplio.rename(columns={2:df_analisis_sentimiento_frente_amplio.loc[0, 2]})
df_analisis_sentimiento_frente_amplio['cant_tot_extraidos']=len(df)
df_analisis_sentimiento_frente_amplio.drop([0],axis=0,inplace=True)
#df_analisis_sentimiento_frente_amplio


# In[42]:


lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = stopwords.words('spanish')


# In[43]:


question_list = list()
for question in df.full_text1:
  question_list.append(text_to_wordlist_wc(str(question).strip()))
df1 = pd.DataFrame(question_list, columns =['full_text1']) 
#df1


# In[44]:


comment_words = '' 
for val in df1.full_text1: 
    val = str(val) 
    # split the value 
    tokens = val.split() 
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += " ".join(tokens)+" "


# In[45]:


tokenized_word=word_tokenize(comment_words)
#print(tokenized_word)


# In[46]:


fdist = FreqDist(tokenized_word)
print(fdist)
xd=fdist.most_common(20)


# In[47]:


# Analisis de tendencia app
df_analisis_tendencia_frente_amplio = pd.DataFrame(data=xd,columns=['termino','frecuencia'])
source_df = df.groupby(('source')).numero.sum()
source_sorted = sorted(source_df.items(), key=lambda x: x[1], reverse=True)
source_t= pd.DataFrame(data=source_sorted,columns=['source','cantidad'])
#df_analisis_tendencia_frente_amplio


# In[48]:


df_t_usuarios_frente_amplio = df.groupby(['created_at','POS']).agg(
                                  {'numero': 'sum'
                                  }).reset_index()

df_t_usuarios_frente_amplio.rename(columns={'created_at': 'fecha', 
                                    'POS': 'sentimiento','numero':'tweet'}, inplace=True)
#df_t_usuarios_frente_amplio


# In[49]:


#conectamos a la base de datos
mongod_connect = 'mongodb://localhost:27017'
client = MongoClient(mongod_connect)
db =client['db_twitter']


# In[50]:


#creamos la tabla analisis_sentimiento_frente_amplio
analisis_sentimiento_frente_amplio = db.analisis_sentimiento_frente_amplio 
records = json.loads(df_analisis_sentimiento_frente_amplio.T.to_json()).values()
db.analisis_sentimiento_frente_amplio.insert_many(records)


# In[51]:


df_t_frente_amplio =pd.DataFrame()
df_t_frente_amplio['id_str']         =df['id_str']
df_t_frente_amplio['screen_name']    =df['screen_name']
df_t_frente_amplio['created_at']     =df['created_at']
df_t_frente_amplio['full_text']      =df['full_text']
df_t_frente_amplio['favorite_count'] =df['favorite_count']
df_t_frente_amplio['retweet_count']  =df['retweet_count']
df_t_frente_amplio['location']       =df['location']
df_t_frente_amplio['source']         =df['source']
df_t_frente_amplio['pos']            =df['POS']


# In[52]:


df_usuario_frente_amplio = db.t_usuario_frente_amplio
records = json.loads(df_t_frente_amplio.T.to_json()).values()
db.t_frente_amplio.insert_many(records)


# In[53]:


analisis_tendencia_frente_amplio = db.analisis_tendencia_frente_amplio
records = json.loads(df_analisis_tendencia_frente_amplio.T.to_json()).values()
db.analisis_tendencia_frente_amplio.insert_many(records)


# In[54]:


t_usuarios_frente_amplio = db.t_usuarios_frente_amplio
records = json.loads(df_t_usuarios_frente_amplio.T.to_json()).values()
db.t_usuarios_frente_amplio.insert_many(records)


# In[ ]:





# In[ ]:




