# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:18:34 2019

@author: Rohit
"""

# from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Flatten, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle
import nltk
from nltk.corpus import stopwords


# =============================================================================
# preprocessing
# =============================================================================
import re
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords

chunksize = 100000
chunks = pd.read_csv('NOTEEVENTS.csv', chunksize=chunksize);
    
dfList = []
for chunk in chunks:
    #df_mimic_chunk = chunk # used with smaller set of data
    #break    
    dfList.append(chunk)

# used with larger set of data
df = pd.concat(dfList,sort=False)
df_mimic_chunk_text = list(df['TEXT'])

for index, val in enumerate(df_mimic_chunk_text): 
    stratingString = df_mimic_chunk_text[index].find("History of Present Illness:")
    endingString = df_mimic_chunk_text[index].find("Physical Exam:")
    if stratingString == -1 or endingString == -1:
        df_mimic_chunk_text[index] = ""
        continue
    df_mimic_chunk_text[index] = re.sub('[^a-zA-Z]', ' ', str(df_mimic_chunk_text[index]))
    df_mimic_chunk_text[index] = df_mimic_chunk_text[index].lower()
    df_mimic_chunk_text[index] = df_mimic_chunk_text[index].split()
    df_mimic_chunk_text[index] = [word for word in df_mimic_chunk_text[index] if not word in set(stopwords.words('english'))]
    df_mimic_chunk_text[index] = ' '.join(df_mimic_chunk_text[index])
    
with open('/home/chandra/rohit-lstm/lstm-large/data/mimic-data/txt/mimic-data-file.txt', 'w') as filehandle:  
    filehandle.writelines("%s\n" % data for data in df_mimic_chunk_text)


# import spacy, and english model
import spacy
nlp = spacy.load('en')

# setting the parameters
# data directory containing input.txt
# =============================================================================
# data_dir = 'C:\\Masters-LUC\\spring-2019\\research\\mimic-iii-project\\lstm-model\\data\\mimic-data\\txt'
# # directory to store models
# save_dir = 'C:\\Masters-LUC\\spring-2019\\research\\mimic-iii-project\\lstm-model\\save' 
# =============================================================================

# Directory structure for GPU
# data_dir = '/home/chandra/rohit-lstm/lstm-model/data/mimic-data/txt'
# data_dir = '/home/chandra/rohit-lstm/lstm-model/data/Gutenberg/txt'
data_dir = '/home/chandra/rohit-lstm/lstm-large/data/mimic-data/txt'
#save_dir = '/home/chandra/rohit-lstm/lstm-model/save'
save_dir = '/home/chandra/rohit-lstm/lstm-large/save'

seq_length = 30 # sequence length
sequences_step = 1 #step to create sequences


from os import listdir
from os.path import isfile, join
import random
# file_list = random.sample([f for f in listdir(data_dir) if isfile(join(data_dir, f))], 3)
# print(file_list)

#for cpu 
# =============================================================================
# file_list = ['Charles Dickens___The Chimes.txt', 'G K Chesterton___The Trees of Pride.txt']
# =============================================================================

# for gpu
file_list = ['mimic-data-file.txt']
# file_list = ['Charles Dickens___The Chimes.txt', 'G K Chesterton___The Trees of Pride.txt']
vocab_file = join(save_dir, "words_vocab.pkl")


num_words = 0
for txt in file_list:
    with open(data_dir+'/'+txt, 'r') as f:
    # with open(data_dir+'/'+txt, 'r', encoding="utf8") as f:
        for line in f:
            words = line.split()
            num_words += len(words)
print("Number of words:")
print(num_words)

# =============================================================================
# document = nlp(u'My name is rohit jagannath. I am a masters degreee holder. I want to get back to stream')
# print(document)
# print(type(document))
# =============================================================================

# read data
def create_wordlist(doc):
    wl = []
    for word in doc:
        if word.text not in ("\n","\n\n",'\u2009','\xa0'):
            wl.append(word.text.lower())
    return wl

#pre-processing
# =============================================================================
# punctuation = '!@#$%^&*()_-+={}[]:;"\'|<>,.?/~`'
# print(nlp.Defaults.stop_words)
# =============================================================================

# create list of sentences
wordlist = []
for file_name in file_list:
    input_file = os.path.join(data_dir, file_name)
    #read data
    with codecs.open(input_file, "r", encoding="utf8") as f:
        data = f.read()
    #create sentences
    doc = nlp(data)
    wl = create_wordlist(doc)
    wordlist = wordlist + wl    
    
# create dictionary
    
# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)
print(words)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)
    
# create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences:', len(sequences))   

# training
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1
    

# Build Model
def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model

rnn_size = 256 # size of RNN
batch_size = 32 # minibatch size
seq_length = 30 # sequence length
num_epochs = 20 # number of epochs
learning_rate = 0.001 #learning rate
sequences_step = 1 #step to create sequences

md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()

#fit the model
callbacks=[EarlyStopping(patience=4, monitor='val_loss'),
           ModelCheckpoint(filepath=save_dir + "/" + 'my_model_gen_sentences_lstm.{epoch:02d}-{val_loss:.2f}.hdf5',\
                           monitor='val_loss', verbose=0, mode='auto', period=2)]
history = md.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks,
                 validation_split=0.01)

#save the model
md.save(save_dir + "/" + 'my_model_gen_sentences_lstm.final.hdf5')

#load vocabulary
print("loading vocabulary...")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

from keras.models import load_model
# load the model
print("loading model...")
model = load_model(save_dir + "/" + 'my_model_gen_sentences_lstm.final.hdf5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#initiate sentences
seed_sentences = u'eventual'
generated = ''
sentence = []
for i in range (seq_length):
    sentence.append("due")

seed = seed_sentences.split()

for i in range(len(seed)):
    sentence[seq_length-i-1]=seed[len(seed)-i-1]

generated += ' '.join(sentence)
print('Generating text with the following seed: "' + ' '.join(sentence) + '"')

print ()

words_number = 10
#generate the text
for i in range(words_number):
    #create the vector
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    #print(x.shape)

    #calculate next word
    preds = model.predict(x, verbose=0)[0]
    predsList = preds.tolist();
    top5indexes = sorted(range(len(predsList)), key=lambda i: predsList[i], reverse=True)[:10]
    top5words = []
    for i in top5indexes:
        top5words.append(vocabulary_inv[i]);
    print(top5words)
    
    
    print("Predictions: ", preds)
    print("Sorted Predictions: ", predsList)
    next_index = sample(preds, 0.34)
    print("next_index: ", next_index)
    next_word = vocabulary_inv[next_index]
    print("next_word: ", next_word)

    #add the next word to the text
    generated += " " + next_word
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]

print(generated)




