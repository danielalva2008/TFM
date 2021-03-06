# **Analisis del Partido Accion Popular**
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
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import configparser
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

# Descargamos los datos de Twitter
from tweepy import OAuthHandler
# Claves del acceso
consumer_key = 'eMcnfrsrAgghHJCCBdP8QStYj'
consumer_secret = 'M2FwH7ARB8MK6hq0n8HTrVErWanAsTST0nkmw11VfvuMjn8fel'
access_token = '587863980-j6aJNJaXotnCuYF5rBZ8myUYF8kCet61E5sgWWHC'
access_token_secret = 'TbSQ6OVufSk0LH6iiRpSAJBwffggEOR2Ah1ToXRQghRIV'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api=tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True,parser=tweepy.parsers.JSONParser())

# Guardamos los datos en archivo JSON
count = 30000
date_since="2019-10-01"
search_words="#AccionPopular"
api = tweepy.API(auth)                                                                                                 
results = [status._json for status in tweepy.Cursor(api.search, q=search_words, count=count,since=date_since, tweet_mode='extended', lang='es').items()]
results
with open('AccionPopularS1OK.json', 'w') as file:
    json.dump(results, file, indent=4)

twet_data= open('AccionPopularS1OK.json', 'r').read()
twet = json.loads(twet_data)

with open('AccionPopularS1OK.json') as train_file:
    dict_train = json.load(train_file, encoding='latin-1')
tweets = pd.DataFrame.from_dict(dict_train)
len(tweets.index)

# Almacenamos los datos mas relevantes-filtrado y transformacion
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

dfx = pd.DataFrame()
dfx['id_str'] =df['id_str']
dfx['full_text_i'] =df['full_text']
dfx = dfx.reset_index(drop=True)
#Normalizamos el campo fecha
df['created_at'] = pd.to_datetime(df.created_at)
df['created_at'] = df['created_at'].dt.normalize()
df['numero']=1
df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d')
df = df.reset_index(drop=True)

question_list = list()
for question in df.full_text:
  question_list.append(text_to_wordlist(str(question).strip()))
df1 = pd.DataFrame(question_list, columns =['full_text'])
df1['full_text_i']=dfx['full_text_i']


with open('tokenizador/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

from tensorflow.keras.models import load_model
Dcnn2 = load_model("dcnn")
pred = df1.full_text.apply(tokenizer.encode)
pred = tf.keras.preprocessing.sequence.pad_sequences(pred,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=40)
len(np.argmax(Dcnn2.predict(pred),axis=1))

# Analisis de Sentimiento
my_array=np.argmax(Dcnn2.predict(pred),axis=1)
dfp = pd.DataFrame(my_array, columns = ['POS'])
dfp['POS1'] = dfp['POS'].map({0:"Negativo", 1:"Neutro",2:"Positivo"})
df['POS']=dfp['POS1']
df['POS1']=dfp['POS1'].map({"Negativo":"cant_tot_negativos", "Neutro":"cant_tot_neutro","Positivo":"cant_tot_positivos"})
total_app = df.groupby(['POS1']).agg(
                                  {'numero': 'sum'
                                  }).reset_index()


df_analisis_sentimiento_accion_popular = total_app.T
df_analisis_sentimiento_accion_popular.reset_index(drop=True,inplace=True)
df_analisis_sentimiento_accion_popular = df_analisis_sentimiento_accion_popular.rename(columns={0:df_analisis_sentimiento_accion_popular.loc[0, 0]})
df_analisis_sentimiento_accion_popular = df_analisis_sentimiento_accion_popular.rename(columns={1:df_analisis_sentimiento_accion_popular.loc[0, 1]})
df_analisis_sentimiento_accion_popular = df_analisis_sentimiento_accion_popular.rename(columns={2:df_analisis_sentimiento_accion_popular.loc[0, 2]})
df_analisis_sentimiento_accion_popular['cant_tot_extraidos']=len(df)
df_analisis_sentimiento_accion_popular.drop([0],axis=0,inplace=True)
#df_analisis_sentimiento_accion_popular

lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = stopwords.words('spanish')

question_list = list()
for question in df.full_text1:
  question_list.append(text_to_wordlist_wc(str(question).strip()))
df1 = pd.DataFrame(question_list, columns =['full_text1']) 
#df1

comment_words = '' 
for val in df1.full_text1: 
    val = str(val) 
    # split the value 
    tokens = val.split() 
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += " ".join(tokens)+" "

tokenized_word=word_tokenize(comment_words)
#print(tokenized_word)

fdist = FreqDist(tokenized_word)
print(fdist)
xd=fdist.most_common(20)

# Analisis de tendencia app
df_analisis_tendencia_app = pd.DataFrame(data=xd,columns=['termino','frecuencia'])
source_df = df.groupby(('source')).numero.sum()
source_sorted = sorted(source_df.items(), key=lambda x: x[1], reverse=True)
source_t= pd.DataFrame(data=source_sorted,columns=['source','cantidad'])
#df_analisis_tendencia_app

df_t_usuarios_accion_popular = df.groupby(['created_at','POS']).agg(
                                  {'numero': 'sum'
                                  }).reset_index()

df_t_usuarios_accion_popular.rename(columns={'created_at': 'fecha', 
                                    'POS': 'sentimiento','numero':'tweet'}, inplace=True)
#df_t_usuarios_accion_popular

#conectamos a la base de datos
mongod_connect = 'mongodb://localhost:27017'
client = MongoClient(mongod_connect)
db =client['db_twitter']

#creamos las tablas en mongo para el partido Accion Popular
analisis_sentimiento_accion_popular = db.analisis_sentimiento_accion_popular 
records = json.loads(df_analisis_sentimiento_accion_popular.T.to_json()).values()
db.analisis_sentimiento_accion_popular.insert_many(records)

df_t_accion_popular =pd.DataFrame()
df_t_accion_popular['id_str']         =df['id_str']
df_t_accion_popular['screen_name']    =df['screen_name']
df_t_accion_popular['created_at']     =df['created_at']
df_t_accion_popular['full_text']      =df['full_text']
df_t_accion_popular['favorite_count'] =df['favorite_count']
df_t_accion_popular['retweet_count']  =df['retweet_count']
df_t_accion_popular['location']       =df['location']
df_t_accion_popular['source']         =df['source']
df_t_accion_popular['pos']            =df['POS']

df_usuario_accion_popular = db.t_usuario_accion_popular
records = json.loads(df_t_accion_popular.T.to_json()).values()
db.t_accion_popular.insert_many(records)

analisis_tendencia_app = db.analisis_tendencia_app
records = json.loads(df_analisis_tendencia_app.T.to_json()).values()
db.analisis_tendencia_app.insert_many(records)

t_usuarios_accion_popular = db.t_usuarios_accion_popular
records = json.loads(df_t_usuarios_accion_popular.T.to_json()).values()
db.t_usuarios_accion_popular.insert_many(records)