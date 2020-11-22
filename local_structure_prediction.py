#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.vectors import calc_dihedral

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, math

import urllib.request, urllib.error
import nltk

def degrees(rad):
    return (rad * 180) / math.pi

def phi_psi_omega_to_abego(phi, psi, omega):
    #if np.isnan(psi): return ‘O’
    #if np.isnan(omega): omega = 180
    #if np.isnan(phi): phi=90
    
    if np.isnan(phi) or np.isnan(psi) or np.isnan(omega): 
        return 'X'
    
    if abs(omega) < 90:
        return 'O'
    
    elif phi > 0:
        if -100.0 <= psi < 100:
            return 'G'
        else:
            return 'E'
        
    else:
        if -75.0 <= psi < 50:
            return 'A'
        else:
            return 'B'
        
    return 'X'

df = pd.read_csv('2018-06-06-ss.cleaned.csv') # read in CSV

# first 200 pdb ids
input_pdbs = df['pdb_id'][:1000].values.T

# print(input_pdbs)


# In[ ]:


files = []

for pdb in input_pdbs:
    
    try:
        if pdb not in os.listdir((os.getcwd() + '/pdb_files/')):
            filename = urllib.request.urlretrieve('https://files.rcsb.org/download/{}.pdb'.format(pdb), pdb + '.pdb')
            path = os.path.join(os.getcwd(), pdb + '.pdb')
            os.rename(path, os.getcwd() + '/pdb_files/' + pdb + '.pdb') 
            files.append(filename[0])
        else:
            files.append(pdb)
        
    except urllib.error.HTTPError:
        continue    
        
files = set(files)


# In[ ]:


# output set
abegopatterns = []

# input set
seqs = []

cwd = os.getcwd()

for file in files:
        
    phi_psi = []
    nres = []
    
    ran = False
    repeat = True
        
    structure = PDBParser(QUIET = True).get_structure(cwd + "/pdb_files/" + file, cwd + "/pdb_files/" + file)

    for chain in structure:

        polypeptides = PPBuilder().build_peptides(chain)

        for polypeptide in polypeptides:

            ran = True
            
            # a list of polypeptide chain lengths
            nres.append(len(polypeptide))
            
            if len(nres) > 1:
                nres[-1] = nres[-1] + nres[-2]
            
            phi_psi += polypeptide.get_phi_psi_list()  
            
            if polypeptide.get_sequence() not in seqs:
                repeat = False # don't want duplicate sequences
                seqs.append(polypeptide.get_sequence())

            break # only the first subunit for now
            
        break
    
    
    if not(ran) or repeat:
        continue
        
    phi_psi_omega = []

    residues = [res for res in structure.get_residues()]

    for i in range(len(residues) - 1):

        if (i + 1) in nres:
            omega = None
            break

        else:
            try:
                a1 = residues[i]['CA'].get_vector()
                a2 = residues[i]['C'].get_vector()
                a3 = residues[i + 1]['N'].get_vector()
                a4 = residues[i + 1]['CA'].get_vector()
                
                omega = calc_dihedral(a1,a2,a3,a4)
                
                phi_psi_omega.append((phi_psi[i][0], phi_psi[i][1], omega))
                
            except KeyError:
                # phi_psi_omega.append((phi_psi[i][0], phi_psi[i][1], None))
                # seqs.pop()
                continue
    
    # last triplet tuple
    phi_psi_omega.append((phi_psi[-1][0], phi_psi[-1][1], None))
    
    # ABEGO str
    abego = ""

    for phi, psi, omega in phi_psi_omega: 
        if phi != None and psi != None and omega != None:
            abego += phi_psi_omega_to_abego(degrees(phi), degrees(psi), degrees(omega))
        
    abegopatterns.append(abego)


# In[ ]:


print(len(seqs), len(abegopatterns))

for i in range(len(seqs)):
    seqs[i] = str(seqs[i])

def seq2ngrams(seqs, n=3):
    return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs])

input_grams = seq2ngrams(seqs)


# In[ ]:


from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick

from keras.utils import to_categorical

# encoder, turns sequence into a fixed vector of numbers

# hash function is a possibility

tokenizer_encoder = Tokenizer() # Tokenizer Class Instance

tokenizer_encoder.fit_on_texts(input_grams) # tokenize the input_grams, updates internal unique vocabulary

input_data = tokenizer_encoder.texts_to_sequences(input_grams) # assigns the text a number

input_data = pad_sequences(input_data, maxlen=500, padding='post')

# decoder
tokenizer_decoder = Tokenizer(char_level = True) # every character will be treated as a token because it's ABEGO

tokenizer_decoder.fit_on_texts(abegopatterns) 

target_data = tokenizer_decoder.texts_to_sequences(abegopatterns)

target_data = pad_sequences(target_data, maxlen=500, padding='post')

target_data = to_categorical(target_data) # only zeros and ones

input_data.shape, target_data.shape


# In[ ]:


from keras.models import Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional

maxlen_seq = 500

n_words = len(tokenizer_encoder.word_index) + 1 # Number of Possible Amino Acids

n_tags = len(tokenizer_decoder.word_index) + 1 # Possible ABEGO Patterns

input_seq = Input(shape = (maxlen_seq, )) # 500 rows

x = tf.keras.layers.Embedding(input_dim = n_words, output_dim = maxlen_seq, input_length = maxlen_seq)(input_seq)

x = tf.keras.layers.Bidirectional(LSTM(units = 100, return_sequences = True))(x)

# dense means a lot of non zeros
output = tf.keras.layers.TimeDistributed(Dense(n_tags, activation = "softmax"))(x) 

model = Model(input_seq, output)

model.summary()


# In[ ]:


from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
import tensorflow as tf

# model.compile defines the loss function, optimizer, and metrics
# first metric is Keras provided, second metric is custom metric
model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics = ["accuracy"]) 

X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size = .4, random_state=0)
# seq_train, seq_test, target_train, target_test = train_test_split(seqs, abegopatterns, test_size=.4, random_state=0)

model.fit(X_train, y_train, batch_size=500, epochs=10, validation_data=(X_test, y_test), verbose=2)

