#!pip install emoji
import re
import pandas as pd
import nltk
from string import punctuation
import numpy as np
import emoji

DIACRITICAL_VOWELS = [('á','a'), ('é','e'), ('í','i'), ('ó','o'), ('ú','u'), ('ü','u')]
SLANG = [('d','de'), ('[qk]','que'), ('xo','pero'), ('xa', 'para'), ('[xp]q','porque'),('es[qk]', 'es que'),
              ('fvr','favor'),('(xfa|xf|pf|plis|pls|porfa)', 'por favor'), ('dnd','donde'), ('tb', 'también'),
              ('(tq|tk)', 'te quiero'), ('(tqm|tkm)', 'te quiero mucho'), ('x','por'), ('\+','mas'),
              ('piña','mala suerte'),('agarre','adulterio'),('ampay','verguenza'),('bacan','alegria'),
              ('bamba','falsificado'),('cabeceador','ladron'),('cabro','homosexual'),('cachaciento','burlon'),
              ('calabacita','tonta'),('caleta','secreto'),('cabro','homosexual'),('cana','carcel'),
              ('chucha','molestia'),('choro','ladron'),('conchán','conchudo'),('cutra','ilicito'),
              ('dark','horrible'),('lenteja','torpe'),('lorna','tonto'),('mancar','morir'),
              ('monse','tonto'),('lenteja','torpe'),('lorna','tonto'),('mancar','morir'),('piñata','mala suerte') ]

stop_words=['a', 'actualmente', 'adelante', 'además', 'afirmó', 'agregó', 'ahí', 'ahora',
    'cc', 'pa', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'al',
    'algo', 'algún', 'algún', 'alguna', 'algunas', 'alguno', 'algunos',
    'alrededor', 'ambos', 'ampleamos', 'añadió', 'ante', 'anterior', 'antes',
    'apenas', 'aproximadamente', 'aquel', 'aquellas', 'aquellos', 'aqui',
    'aquí', 'arriba', 'aseguró', 'así', 'atras', 'aún', 'aunque', 'ayer',
    'bajo', 'bastante', 'bien', 'buen', 'buena', 'buenas', 'bueno', 'buenos',
    'cada', 'casi', 'cerca', 'cierta', 'ciertas', 'cierto', 'ciertos', 'cinco',
    'comentó', 'como', 'cómo', 'con', 'conocer', 'conseguimos', 'conseguir',
    'considera', 'consideró', 'consigo', 'consigue', 'consiguen', 'consigues',
    'contra', 'cosas', 'creo', 'cual', 'cuales', 'cualquier', 'cuando',
    'cuanto', 'cuatro', 'cuenta', 'da', 'dado', 'dan', 'dar', 'de', 'debe',
    'deben', 'debido', 'decir', 'dejó', 'del', 'demás', 'dentro', 'desde',
    'después', 'dice', 'dicen', 'dicho', 'dieron', 'diferente', 'diferentes',
    'dijeron', 'dijo', 'dio', 'donde', 'dos', 'durante', 'e', 'ejemplo', 'el',
    'de', 'la', 'el', 'porfas', 't', 'p', 'd', 'est',
    'él', 'ella', 'ellas', 'ello', 'ellos', 'embargo', 'empleais', 'emplean',
    'emplear', 'empleas', 'empleo', 'en', 'encima', 'encuentra', 'entonces',
    'entre', 'era', 'eramos', 'eran', 'eras', 'eres', 'es', 'esa', 'esas',
    'ese', 'eso', 'esos', 'esta', 'ésta', 'está', 'estaba', 'estaban',
    'estado', 'estais', 'estamos', 'estan', 'están', 'estar', 'estará',
    'estas', 'éstas', 'este', 'éste', 'esto', 'estos', 'éstos', 'estoy',
    'estuvo', 'ex', 'existe', 'existen', 'explicó', 'expresó', 'fin', 'fue',
    'fuera', 'fueron', 'fui', 'fuimos', 'gran', 'grandes', 'gueno', 'ha',
    'haber', 'había', 'habían', 'habrá', 'hace', 'haceis', 'hacemos', 'hacen',
    'hacer', 'hacerlo', 'haces', 'hacia', 'haciendo', 'hago', 'han', 'hasta',
    'hay', 'haya', 'he', 'hecho', 'hemos', 'hicieron', 'hizo', 'hoy', 'hubo',
    'igual', 'incluso', 'indicó', 'informó', 'intenta', 'intentais',
    'intentamos', 'intentan', 'intentar', 'intentas', 'intento', 'ir', 'junto',
    'la', 'lado', 'largo', 'las', 'le', 'les', 'llegó', 'lleva', 'llevar',
    'lo', 'los', 'luego', 'lugar', 'manera', 'manifestó', 'más', 'mayor', 'me',
    'mediante', 'mejor', 'mencionó', 'menos', 'mi', 'mientras', 'mio', 'misma',
    'mismas', 'mismo', 'mismos', 'modo', 'momento', 'mucha', 'muchas', 'mucho',
    'muchos', 'muy', 'nada', 'nadie', 'ni', 'ningún', 'ninguna', 'ningunas',
    'ninguno', 'ningunos', 'nos', 'nosotras', 'nosotros', 'nuestra',
    'nuestras', 'nuestro', 'nuestros', 'nueva', 'nuevas', 'nuevo', 'nuevos',
    'nunca', 'o', 'ocho', 'otra', 'otras', 'otro', 'otros', 'para', 'parece',
    'parte', 'partir', 'pasada', 'pasado', 'pero', 'pesar', 'poca', 'pocas',
    'poco', 'pocos', 'podeis', 'podemos', 'poder', 'podrá', 'podrán', 'podria',
    'podría', 'podriais', 'podriamos', 'podrian', 'podrían', 'podrias',
    'poner', 'por', 'porque', 'por qué', 'posible', 'primer', 'primera',
    'primero', 'primeros', 'principalmente', 'propia', 'propias', 'propio',
    'propios', 'próximo', 'próximos', 'pudo', 'pueda', 'puede', 'pueden',
    'puedo', 'pues', 'que', 'qué', 'quedó', 'queremos', 'quien', 'quién',
    'quienes', 'quiere', 'realizado', 'realizar', 'realizó', 'respecto',
    'sabe', 'sabeis', 'sabemos', 'saben', 'saber', 'sabes', 'se', 'sea',
    'sean', 'según', 'segunda', 'segundo', 'seis', 'señaló', 'ser', 'será',
    'serán', 'sería', 'si', 'sí', 'sido', 'siempre', 'siendo', 'siete',
    'sigue', 'siguiente', 'sin', 'sino', 'sobre', 'sois', 'sola', 'solamente',
    'solas', 'solo', 'sólo', 'solos', 'somos', 'son', 'soy', 'su', 'sus',
    'tal', 'también', 'tampoco', 'tan', 'tanto', 'tendrá', 'tendrán', 'teneis',
    'tenemos', 'tener', 'tenga', 'tengo', 'tenía', 'tenido', 'tercera',
    'tiempo', 'tiene', 'tienen', 'toda', 'todas', 'todavía', 'todo', 'todos',
    'total', 'trabaja', 'trabajais', 'trabajamos', 'trabajan', 'trabajar',
    'trabajas', 'trabajo', 'tras', 'trata', 'través', 'tres', 'tuvo', 'tuyo',
    'tu', 'te', 'pq', 'mas', 'qie', 'us', 'has', 'ti', 'ahi', 'mis', 'tus',
    'do', 'X', 'Ven', 'mo', 'Don', 'dia', 'PT', 'sua', 'q', 'x', 'i', 
    'última', 'últimas', 'ultimo', 'último', 'últimos', 'un', 'una', 'unas',
    'uno', 'unos', 'usa', 'usais', 'usamos', 'usan', 'usar', 'usas', 'uso',
    'usted', 'va', 'vais', 'valor', 'vamos', 'van', 'varias', 'varios', 'vaya',
    'veces', 'ver', 'verdad', 'verdadera', 'verdadero', 'vez', 'vosotras',
    'n', 's', 'of', 'c', 'the', 'm', 'qu', 'to', 'as', 'is',
    'asi', 'via', 'sera', 'tambien', 'vosotros', 'voy', 'y', 'ya', 'yo','https','rt ']

SLANG_SP_SN = [('[qk]','que'), ('xo ','pero'), ('[xp]q','porque'),('es[qk]', 'es que'),
              ('fvr ','favor'),('(xfa |xf |pf |plis |pls |porfa )', 'por favor'), ('dnd','donde'), 
              ('(tq |tk )', 'te quiero'), ('(tqm |tkm )', 'te quiero mucho'), ('\+','mas'),
              ('piña','mala suerte'),('agarre','adulterio'),('ampay','verguenza'),('bacan','alegria'),
              ('bamba','falsificado'),('cabeceador','ladron'),('cabro','homosexual'),('cachaciento','burlon'),
              ('calabacita','tonta'),('caleta','secreto'),('cabro','homosexual'),('cana','carcel'),
              ('chucha','molestia'),('choro','ladron'),('conchán','conchudo'),('cutra','ilicito'),
              ('dark','horrible'),('lenteja','torpe'),('lorna','tonto'),('mancar','morir'),
              ('monse','tonto'),('lenteja','torpe'),('lorna','tonto'),('mancar','morir'),                 ('piñata','malasuerte'),(':beaming_face_with_smiling_eyes:','agradecimiento'), (':grinning_face_with_big_eyes:','alegria'), (':grinning_face_with_sweat:','optimismo'), (':grinning_squinting_face:','risa'),
(':flexed_biceps:','esfuerzo'),(':green_heart:','amor'),(':heart_with_ribbon:','amor'),(':smiling_face_with_halo:','angel'),
(':smiling_face_with_heart-eyes:','Gusto'),(':relieved_face:','Relajo'),(':kissing_face_with_smiling_eyes:','Felicidad'),(':winking_face_with_tongue:','Guiño'),(':grinning_cat_with_smiling_eyes:','Alegria'),(':grinning_cat_with_smiling_eyes:','Alegria'),(':cat_with_tears_of_joy:','Alegria'),(':smiling_cat_with_heart-eyes:','alegria'),
(':raising_hands:','exito'),(':clapping_hands:','admiracion'),(':victory_hand:','aprobacion'),(':check_box_with_check:','aprobacion'),
(':thumbs_down:','desaprobacion'),(':pile_of_poo:','asco'),(':frowning_face:','tristeza'),(':face_screaming_in_fear:','asustado'),(':fearful_face:','temor'),(':loudly_crying_face:','llorar'),(':anxious_face_with_sweat:','preocupacion'),
(':crying_face:','llorar'),(':crying_face:','llorar'),(':downcast_face_with_sweat:','estres'),(':hushed_face:','triste'),(':confused_face:','insatisfecho'),(':frowning_face:','insatisfecho'),
 (':worried_face:','preocupacion'),(':disappointed_face:','decepcion'),(':pensive_face:','pensativo'),(':unamused_face:','aburrido'),(':persevering_face:','perseverante'),(':confounded_face:','consternacion'), (':weary_face:','cansado'),(':pouting_face:','enfurecer'),(':angry_face:','enfado'),(':angry_face_with_horns:','enfado'),(':broken_heart:','decepcion'),(':loudly_crying_face:','derrota'),(':weary_cat:','cansado'),(':crying_cat:','llorar'),(':cross_mark:','desaprobacion')]

stop_words_p= ['https']


def eliminarhtml(text):
    text = str(text).strip()
    text = re.sub(r'<[^>]*?>', '', text) 
    return(text)


def encoding_emoji(df):
	for i in df.index:
		string=''
		for w in df.loc[i,'tweet'].split():
			W=emoji.demojize(w)
			string=string+W+' '
		df.loc[i,'tweet']=string
	return df

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # limpiar el  texto
    text = str(text).strip()
    #remplazar los acentos
    for s,t in DIACRITICAL_VOWELS:
        text = re.sub(r'{0}'.format(s), t, text)
   #remplazar el SLANG
    for s,t in SLANG:
        text = re.sub(r'\b{0}\b'.format(s), t, text)
        
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
    text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames
    text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag

    text = text.lower()
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # remover stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    #if stem_words:
    #    text = text.split()
    #    stemmer = SnowballStemmer('spanish')
    #    stemmed_words = [stemmer.stem(word) for word in text]
    #    text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

def text_to_wordlist_wc(text, remove_stop_words=True, stem_words=False):
    # limpiar el  texto
    text = str(text).strip()
    #remplazar los acentos
    for s,t in DIACRITICAL_VOWELS:
        text = re.sub(r'{0}'.format(s), t, text)
    #remplazar el SLANG
    #for s,t in SLANG_SP_SN:
    #    text = re.sub(r'{0}'.format(s), t, text)
    #Limpieza de datos
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
    text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames
    text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
  #convertir a minusculas
    text = text.lower()
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    # remover stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words and not w in stop_words_p and len(w)>2]
        text = " ".join(text)
    text = re.sub(r' {2,}' , ' ', text)       
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('spanish')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return(text)