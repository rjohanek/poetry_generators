# AI-Powered Unique Poetry Generators

## About the project
There are two models. 
Naive Bayes model (flexible n-gram implementation) and Keras Sequential Feedforward Neural Network model (trigram implementation).

Written by Christina Pathrose and Rachel Johanek as park of CS4150: Natural Language Processing at Northeastern University.

The Dataset used for this Project features Poems from the Poetry Foundation, and can be found here: 
https://www.kaggle.com/tgdivy/poetry-foundation-poems

The pretrained word embeddings are too large to include here find "GoogleNews-vectors-negative300.bin.gz" here: https://code.google.com/archive/p/word2vec/ under the "Pre-trained word and phrase vectors" section.
Include "GoogleNews-vectors-negative300.bin.gz" in your development folder.

## First, train.
### Naive Bayes:
To train use the following command format:
 ./naive_bayes_model.py INT
  
where...
  INT is an optional parameter specifying the desired n-gram length (default=3)
    
AND
  PoetryFoundationData.csv (Poetry Dataset) must be in the folder
    
### Neural Network:
    To train use the following command format:
     ./keras_model.py 
    
AND
   PoetryFoundationData.csv (Poetry Dataset) must be in the folder
   Pretrained Word Embeddings Original file (GoogleNews-vectors-negative300.bin.gz) must be in the folder

## Then, generate!
To run use the following command format:
 ./poem_generator.py TOPIC INT MODEL_TYPE
  
where...
TOPIC is a String which represents the topic of teh poem (ex. water)
INT is an interger greater than 0, which represents the number of lines in the poem 
MODEL_TYPE is either "NB" or "Keras" (default=Keras), which represents the model you wish to use

AND
   this requires the following files that were generated during training: 
1. PreTrained Naive Bayes Model (my_clf.joblib)
2. Pretrained Keras Sequential Model (poem_words)
3. Pretrained Word Embeddings (pretrained_embeddings.model)
4. Word Embeddings trained on Poetry Dataset (poem_words.model)
