# import your libraries here
import nltk
from gensim.models import Word2Vec
import re
import gensim
import codecs
import pandas as pd
import csv 

# Importing utility functions from Keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Embedding
from keras import metrics

################ README - NEURAL NETWORK MODEL IMPLEMENTATION ###################
"""
Authors: Rae Johanek & Christina Pathrose
CS4120 Final Project 

This file should be used to build the Keras Sequential Model. To run use the following command format:
 ./keras_model.py 

The Dataset used for this Project features Poems from the Poetry Foundation, and can be found here: 
https://www.kaggle.com/tgdivy/poetry-foundation-poems

This file requires the following : 
1. Poetry Dataset (PoetryFoundationData.csv)
2. Pretrained Word Embeddings Original file (GoogleNews-vectors-negative300.bin.gz)

"""
###########################  IMPLEMENTATION  ######################################

# CONSTANTS: 
NGRAM = 3 # Define the size of the ngram language model you want to train: 
EMBEDDINGS_SIZE = 200

# PRE-PROCESSING: 
START = "<s>"
END = "</s>"
sent_delim=" . "
line_delim="\n"

def tokenize(sentences):
    """
    break up the given sentences into tokens within the sentence and normalize the text
    """
    sentences_tokenized = []
    # lowercase the text, remove unwanted punctuation, tokenize each sentence
    # and add the list of tokens to the list of tokenized sentences
    for sent in sentences :
        # pre-process to remove punctuation
        sent_no_punc = re.sub('[!?.,;]', '', sent)
        # tokenize
        sent_tokens = sent_no_punc.lower().split()  
        # add start and end tokens
        sent_tokens.insert(0, START)     
        sent_tokens.append(END)
        # add the list of tokens to the list of sentences
        sentences_tokenized.append(sent_tokens)
    return sentences_tokenized

def pre_pend(sentences):
    """ Add another sentence start token to the beginning of each sentence"""
    pre_pended = []
    for sentence in sentences:
        new = sentence.copy()
        new.insert(0, START)
        pre_pended.append(new)
    return pre_pended

# read in data from csv file
def pre_process(filePath, n):
    data=pd.read_csv(filePath)
    poems = []
    for row in data.iterrows():
        # row[0] is the id
        # row[1] is the info: index at [0], title at [1], poem at [2], poet at [3], and tags at [4]
        poems.append(row[1][2])

    poems_joined_sentences = sent_delim.join(poems)
    poems_joined_lines = line_delim.join(poems)

    # example of line: Dog bone, stapler,
    lines = tokenize(poems_joined_sentences.split('\n'))
    lines = [x for x in lines if len(x) > 2]

    # example of sentence: Dog bone, stapler, cribbage board, garlic press because this window is loose—lacks suction, lacks grip
    sentences = tokenize(re.split('[.!?]', poems_joined_lines))
    sentences = [x for x in sentences if len(x) > 2]
    
    return sentences
  

# Initializing a Tokenizer
# It is used to vectorize a text corpus. Here, it just creates a mapping from word to a unique index. 
# (Note: Indexing starts from 0)
# given a text returns a word to index map. The input is a list of tokenized sentences
def encode_text(text):
    tokenizer = Tokenizer()
    # generate sequences for all words in text
    tokenizer.fit_on_texts(text)
    encoded = tokenizer.texts_to_sequences(text)
    
    # get word to sequence map
    word_to_encoded_map = tokenizer.word_index
    encoded_to_word_map = {k:v for k,v in zip(word_to_encoded_map.values(), word_to_encoded_map.keys())}
        
    return (word_to_encoded_map, encoded_to_word_map)

def make_n_grams(tokens): 
    """Creates tuples of n length representing the n-grams in the given tokens
    Parameters:
      tokens (list of str): the list of individual tokens to be grouped into n-grams

    Returns:
      n_grams (list of tuples of str): the list of n-grams created from the given tokens
    """ 
    n_grams = []

    if NGRAM == 1 :
        return tokens
    else:
        for i in range(NGRAM - 1, len(tokens)):
            n_gram = [tokens[j] for j in range(i - (NGRAM - 1), i + 1)]
            n_grams.append(tuple(n_gram))
        return n_grams

def generate_training_samples(word_to_encoded, data): 
    '''
    Takes the encoded data map of word to index and 
    generates the training samples out of it.
    Parameters:
    encoded data map of word to index
    return: 
    tuple of lists in the format ([all n-1 grams with last words excluded], [all last words of the n-grams])
    '''
    # list of all n-grams for the text in the encoded list
    data_as_one_list =  [token for sublist in data for token in sublist]
    all_ngrams = make_n_grams(data_as_one_list)
    all_n_minus_1_grams = []
    all_last_word_in_grams = []
    
    for ngram in all_ngrams:
        all_n_minus_1_grams.append(ngram[:-1])
        all_last_word_in_grams.append(ngram[-1])
        
    return [all_n_minus_1_grams, all_last_word_in_grams]


def read_embeddings():
    '''Loads and parses embeddings trained in earlier.
    Parameters and return values are up to you.
    '''
    # word to embedding maps
    wv_poem = model_poems.wv
    return wv_poem

def pre_process_words_to_embeddings(embeddings, words):
    """ Conver word lists to a new concatenated embedding represnting the embedding of each word"""
    word_embeddings = []
    for word in words:
        word_embeddings.append((embeddings[word]).ravel())
    return np.concatenate(word_embeddings)

def pre_pend(sentences):
    """ Add another sentence start token to the beginning of each sentence"""
    pre_pended = []
    for sentence in sentences:
        new = sentence.copy()
        new.insert(0, "<s>")
        pre_pended.append(new)
    return pre_pended

################ DATAGENERATOR CLASS ###################
class DataGenerator(Sequence):
   
    'Generates data for Keras'
    def __init__(self, X, y, batch_size, embeddings, word_to_encoded):
        'Initialization'
        self.batch_size = num_sequences_per_batch
        self.outputs = y
        self.inputs = X
        self.embeddings = embeddings
        self.word_to_encoded = word_to_encoded
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of X and y values
        list_X = [self.inputs[j] for j in indexes]
        list_y = [self.outputs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_X, list_y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.inputs))
    
    def __data_generation(self, X: list, y: list) -> (list,list):
        #X : n-1 grams
        #y: last token in the gram
    
        vocab_size = len(self.word_to_encoded)
        n_1_grams_embeddings = np.empty((self.batch_size, EMBEDDINGS_SIZE*(NGRAM - 1)))
        last_tokens = np.empty((self.batch_size))
    
        for i in range(len(X)):
            words = X[i]
            word_embeddings = []
            for word in words:
                word_embeddings.append((self.embeddings[word]).ravel())
            concatenated = np.concatenate(word_embeddings)
        
            n_1_grams_embeddings[i] = concatenated
            last_tokens[i] = self.word_to_encoded[y[i]]
            
        one_hot = to_categorical(last_tokens, vocab_size)
        
        return n_1_grams_embeddings, one_hot
    
    '''
    Returns data generator to be used by feed_forward
    https://wiki.python.org/moin/Generators
    https://realpython.com/introduction-to-python-generators/
    
    Yields batches of embeddings and labels to go with them.
    Use one hot vectors to encode the labels 
    (see the to_categorical function)
    
    '''

# makes the neural network language model 
def make_neural_language(embedding_matrix):
    model = Sequential()
    # hidden layer
    model.add(Dense(500, activation="relu", input_dim=EMBEDDINGS_SIZE*(NGRAM - 1)))
    # output layer
    model.add(Dense(len(embedding_matrix), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
    print(model.summary())
    return model

# generate a sequence from the model
def generate_seq(model: Sequential, 
                 embeddings,
                 seed: list, 
                 length: int):
    '''
    Parameters:
        model: your neural network
        tokenizer: the keras preprocessing tokenizer
        seed: [w1, w2, ... w(m-1)]
        length: generate a sentence of length n_words
    Returns: string sentence
    '''
    # seed: initial list of words or could just be [<s>]
    # generate next word(s) in sequence
    # given current sequence, use model to predict what comes next
    # take input, run it through the model (apply nonlinearity), 
    starting = np.expand_dims(pre_process_words_to_embeddings(embeddings, seed), axis=0)
    
    previous_word = seed[1]
    sentence = []
    while len(sentence) < length:
        prob_dist = model.predict(starting).ravel()
        new_word = np.random.choice(embeddings.index_to_key, p=prob_dist)
        
        # don't include sentence end tokens
        while new_word == ("/n") or new_word == ("<s>"):
            new_word = np.random.choice(embeddings.index_to_key, p=prob_dist)
            
        sentence.append(new_word)
        starting = np.expand_dims(pre_process_words_to_embeddings(embeddings, [previous_word, new_word]), axis=0)
        previous_word = new_word
        
    return " ".join(sentence)

# generates sentences given model, embeddings, number of sentences, sentence length: 
def generate_sentences(model, embeddings, num_sentences=50, sentence_length=20):
    """ create the the specified number of sentences at the specified length"""
    sentences = []
    for i in range(num_sentences):
        sentences.append(generate_seq(model, embeddings, ["splashing", "boiling"], sentence_length))
    return sentences


######################## main method  for training ##################################


def main():
    """
    Train and save the Neural Network Model 
    """
    # PRE-PROCESS the training data: 
    training_data = pre_process("PoetryFoundationData.csv", 3)

    # Create Word Embeddings with Training Data: 
    model_poems = Word2Vec(sentences=training_data, vector_size=EMBEDDINGS_SIZE, window=5, min_count=1, sg=1)
    # save Word2Vec model: 
    model_poems.save("word2vec.model")

    # print out the vocabulary size
    #print('Vocab size {}'.format(len(model_poems.wv)))

    # PRE-TRAINED WORD EMBEDDINGS:
    W2V_PATH="GoogleNews-vectors-negative300.bin.gz"
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
    model_w2v.save("pretrained_embeddings.model")


    (word_to_encoded, encoded_to_word) = encode_text(training_data) 
    

    # [X: n-1 gram word to encoding map entries (lists), y: nth tokenword to encoding map entries (strings)]
    samples = generate_training_samples(word_to_encoded, training_data) 
    embeddings = read_embeddings()
    num_sequences_per_batch = 1024 #1024 is new batch size

    steps_per_epoch = len(samples[0])//num_sequences_per_batch  # Number of batches per epoch
    #print(steps_per_epoch)

    train_generator = DataGenerator(samples[0], samples[1], num_sequences_per_batch, embeddings, word_to_encoded)

    sample1 = train_generator.__getitem__(0) # this is how you get data out of generators
    print(sample1[0].shape) # (batch_size, (n-1)*EMBEDDING_SIZE)  (128, 200)

    # code to create a feedforward neural language model 
    # with a set of given word embeddings
    nn_model_poems = make_neural_language(embeddings)

    # Start training the model
    nn_model_poems.fit(train_generator, 
          steps_per_epoch=steps_per_epoch,
          epochs=1)

    #nn_model_poems.save("./poems_model")



if __name__ == '__main__':

    main()
