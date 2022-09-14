# -*- coding: utf-8 -*-
"""
This script reads in O*NET occupation task descriptions, computes
pairwise similarity scores, and writes them to file 
"""

#Importing necessary libraries 
import nltk 
import pandas as pd
import numpy as np 
import string
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.keyedvectors import KeyedVectors
from gensim import matutils
import re
from sklearn.metrics.pairwise import cosine_similarity

#Get list of stop words
#This document contains our list of stop words. We compiled from various online
#stop words lists. See Kogan, Papanikolaou, Schmidt, and Seegmiller (2021) for 
#the specific sources. 
stop_words = list(pd.read_csv("stop_words.csv")) 
stop = set(stop_words) #A set is faster for looking up 
    
#Functions to clean text data
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_consecutive_spaces(text):
    return re.sub(' +', ' ', text)

str_list = ['0','1','2','3','4','5','6','7','8','9']
def clean_text(text):
    cleaned = text 
    for a in str_list:
            cleaned = cleaned.str.replace(a,' ')
    cleaned = cleaned.apply(remove_punctuation)
    cleaned = cleaned.apply(remove_non_ascii)
    cleaned = cleaned.map(lambda x: x.lower())
    return cleaned

#These are the ways the NLTK part of speech tagger will label a word if it is a noun or verb
pos_list = set(['NN','NNS','NNP','NNPS', 'VB','VBD','VBG','VBN','VBP','VBZ'])
def get_noun_verbs(x):
    a = [item[0] for item in x if item[1] in pos_list]
    return a

def list_to_string(x):
    return  " ".join([word for word in x])

def remove_stop(x):
    return [item for item in x if item not in stop]

task_filepath ='Task Statements.xlsx'

#Task Statements are at 8 digit SOC level. This grabs the corresponding list of 6 digit codes
def get_soc6():
    tasks = pd.read_excel(task_filepath)
    soc_colname = 'O*NET-SOC Code'  
    
    tasks['soc6'] = tasks[soc_colname].apply(remove_punctuation)
    tasks['soc6'] = tasks['soc6'].apply(lambda x: x[:-2])
    return list(tasks.groupby('soc6')['soc6'].first())

#This function will apply the text cleaning functions above to the task descriptions 
def clean_tasks():
    tasks = pd.read_excel(task_filepath)
    soc_colname = 'O*NET-SOC Code'    
    task_colname = 'Task' 
    
    tasks['soc6'] = tasks[soc_colname].apply(remove_punctuation) #Making sure SOC is just numeric
    tasks['soc6'] = tasks['soc6'].apply(lambda x: x[:-2]) #from 8-digit to 6-digit SOC 
    task = tasks.groupby('soc6')[task_colname].apply(lambda x: '  '.join(x)) #combining all 6-digit task descriptions
    task = task.fillna('')
    task = clean_text(task) #Stripping punctuation and non-alphabetic characters
    task = task.apply(nltk.word_tokenize) #splitting string into lists of words
    task = task.apply(remove_stop) #Removing undesired words 
    task = task.apply(nltk.tag.pos_tag) #Tag part of speech
    task = task.apply(get_noun_verbs) #Retain nouns and verbs 
    task = task.apply(lambda x: [WordNetLemmatizer().lemmatize(word,'v') for word in x] ) #convert verbs to present tense
    task = task.apply(lambda x: [WordNetLemmatizer().lemmatize(word,'n') for word in x] ) #convert nouns to singular form
    return task

print("Cleaning text and fitting tf-idf model...")
taskclean = clean_tasks()

#Creating tf-idf weights for each word in each task description 
dcttask = Dictionary(taskclean)  # fit dictionary
corpustask = [dcttask.doc2bow(line) for line in taskclean]  # convert corpus to BoW format
modeltask = TfidfModel(corpustask)  # fit model

#This function will read in individual word vectors from the Pennington et. al 
#GloVe model of word embeddings and return the weighted average as the document vector
def get_document_vector(arglist):
    '''
    Inputs:
    arglist, a list with 6 elements as follows:
    index--integer, the desired index from tf-idf model corresponding to patent or task
    corpus--documents in doc2bow format based off some gensim dictionary
    tfidfmodel--gensim tf-idf model fit on entire set of documents
    terms--list of terms from dictionary
    wordembeddings--a model containing word embeddngs i.e. a glove/word2vec model
    vocab--the vocabulary from wordembeddings (i.e. set(word_embeddings.vocab))
    Returns:
    x--a tf-idf weighted average document word vector
    '''
    index = arglist[0]
    corpus = arglist[1]
    tfidfmodel = arglist[2]
    terms = arglist[3]
    wordembeddings = arglist[4]
    vocab = arglist[5]
    
    vector = tfidfmodel[corpus[index]]
    indices = [x[0] for x in vector] #Get word corresponding to each tf-idf element
    doctfidf = [x[1] for x in vector] #Get tf-idf score for each element of given document
    desired_terms = [terms[i] for i in indices] #Get list of desired terms
    #Calculate tf-idf weighted document vector
    x = sum([doctfidf[i]*wordembeddings.word_vec(desired_terms[i]) \
             for i in range(len(desired_terms)) if desired_terms[i] in vocab ])
    weight_sum = sum([doctfidf[i] for i in range(len(desired_terms)) if desired_terms[i] in vocab])
    
    if weight_sum > 0:
        x = x/weight_sum
        return np.array(x)
    else:
        return np.nan

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

print("Reading in word vectors...") 
#Load in word vectors from Pennington et al
#Each word vector has 300 elements 
glove_path = 'glove.bin'
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=True)
glove_model = glove_model.wv #Accessing the "wv" attribute works for all models (glove,word2vec, fasttext)
vocab = set(glove_model.vocab)

#List of terms that show up in the union of task descriptions 
taskterms = [item[0] for item in dcttask.token2id.items()]

print("Computing document vectors...")

#Loop over occupations to get document vectors 
vlist = []
#NOTE: Use a simple for loop for simplicity. This can easily be parallelized
#for large computations 
for i in range(len(taskclean)):
    vector = get_document_vector([i,corpustask,modeltask,taskterms,glove_model,vocab])
    vlist = vlist + [vector]    
   
print("Computing Similarity scores and saving data...")
#Convert list of occupation vectors into matrix 
v_mat = np.array([vlist[i] for i in range(len(vlist))]) #NX300 matrix, where N=# of SOC codes
#Very fast way to do pairwise cosine similarities 
sim_mat = cosine_similarity(v_mat,v_mat)  #NXN matrix 

#Write matrix to DataFrame and save off 
soc6_list = get_soc6()
soc6_matrix = np.array([soc6_list]).T.astype(int) #NX1 matrix of SOC codes in integer format 
#Matrix is soc codes in columns and in rows. Label these columns 'occ_codeXXXXXX'
colnames = ['soc' ] + ['occ_code' + x for x in soc6_list]
sim_mat_df = pd.DataFrame(np.hstack((soc6_matrix,sim_mat)),columns=colnames) #N X N + 1 
        
#Write matrix to file
sim_mat_df.to_stata('PairwiseOccSimilarityONET.dta',write_index=False)
 
#------------------------------------------------------------------------------   
#Now compare with TF-IDF weighted bag of words
#------------------------------------------------------------------------------
#This will transform the TF-IDF model to a sparse matrix 
#where we can use the cosine_similarity function to compute similarity scores  
bow_tfidf_mat = matutils.corpus2csc(modeltask[corpustask]).T
bow_tfidf_sim_mat = cosine_similarity(bow_tfidf_mat,bow_tfidf_mat)

bow_tfidf_sim_mat_df = pd.DataFrame(np.hstack((soc6_matrix,bow_tfidf_sim_mat)),columns=colnames) #N X N + 1 
        
#Write matrix to file
bow_tfidf_sim_mat_df.to_stata('PairwiseOccSimilarityONET_TfidfBow.dta',write_index=False)
