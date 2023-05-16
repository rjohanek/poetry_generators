import pandas as pd
import random
import sys
from joblib import load
from tensorflow import keras
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim
from naive_bayes_model import LanguageModel
from pre_processing import pre_process
from art import *

################ README ###################
"""
Authors: Rae Johanek & Christina Pathrose
CS4120 Final Project 

This file should be used to generate our Poem Art. To run use the following command format:
 ./poem_generator.py <TOPIC> <INT> <MODEL_TYPE>
where...
TOPIC is a String which represents the topic of teh poem (ex. water)
INT is an interger greater than 0, which represents the number of lines in the poem 
MODEL_TYPE is either "NP" or "Keras" (default is Keras), which represents the model you wish to use


The Dataset used for this Project features Poems from the Poetry Foundation, and can be found here: 
https://www.kaggle.com/tgdivy/poetry-foundation-poems

This file requires the following models: 
1. PreTrained Naive Bayes Model (my_clf.joblib)
2. Pretrained Keras Sequential Model (poem_words)
3. Pretrained Word Embeddings (pretrained_embeddings.model)
4. Word Embeddings trained on Poetry Dataset (poem_words.model)

"""

################ text formatting ###################

# Formatting method for Naive Bayes model


def make_lines(num_start_tokens, text):
    """
    num_start_tokens: number of start tokens to remove from beginning
    text: list of sentences where a sentence is a list of tokens
    returns: list of strings
      not including start and end tokens with new line characters at the end of every string
    """
    result = []
    for sentence in text:
        # remove start and end
        # # convert from list to string
        result.append(" ".join(sentence[num_start_tokens:-1]))
        # convert from list to string with new lines
    return "\n".join(result)


# method to randombly generate word art Title as part of poem:


def create_title(title):
    tprint(title, font="rnd-xlarge")
    print(text)


# REMOVED:
"""
import cv2
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

# read in image: 
image = cv2.imread("water_background2.jpeg")
# create image mask image: 
#v2.imshow("waterimage", image)
# First, transpose wordCloud onto mask: 
wordcloud = WordCloud(background_color='black', mask=image, mode="RGB", color_func=lambda *args, 
                      **kwargs: "white", width=1000 , max_words=100, height=1000, 
                      random_state=1).generate(text)
# plot: 
fig = plt.figure(figsize=(25,25))
plt.imshow(wordcloud, interpolation='bilinear')
plt.tight_layout(pad=0)
plt.axis("off")
plt.show()
# ERROR: unable to create mask that looks like input image 
"""

############### main method ###################


def main():
    """Receives command line arguments that must be in the following form:
    python3 final_project.py topic length(optional) model(optional)
      where topic is one word
      length is an int
      model is one of NB or keras

    Any input outside of this format will error
    """
    # get topic of poem
    topic = sys.argv[1]

    # get length of poem in lines
    length = 11
    if len(sys.argv) > 3:
        length = int(sys.argv[2])

    # which model will generate the poem
    model = "NB"
    if len(sys.argv) >= 4:
        model = sys.argv[3]

    # get at least 25 similar words, or more if there are more lines
    topn = length if length > 25 else 25

    # create Word2Vec model:
    W2V_PATH = "GoogleNews-vectors-negative300.bin.gz"
    pre_trained_embedding = gensim.models.KeyedVectors.load_word2vec_format(
        W2V_PATH, binary=True)
    print("pre-trained embeddings loaded")

    # load pre-trained word embeddings
    # pre_trained_embedding = # KeyedVectors.load_word2vec_format( datapath("C:\Users\12162\Desktop\NLPtake2\final_project\pretrained_embeddings.model"), binary=False)

    similar_words = pre_trained_embedding.most_similar(topic, topn)
    # shuffle seeds to further randomize poem output:
    random.shuffle(similar_words)

    # generate sentences for trigram model

    # MODEL 1: NAIVE BAYES
    if model == "NB":
        training_data_tri = pre_process("PoetryFoundationData.csv", 3)
        print("done pre-processing data")

        lm_trigram = LanguageModel(3, False)
        lm_trigram.train(training_data_tri)
        print("done training trigram model")

        # verify the model knows the similar words, remove any it doesn't know
        similar_words = lm_trigram.verify_words(similar_words)

        # if the model doesn't know enough of the topic related words to generate the poem
        # generate more
        while (len(similar_words)) < length:
            # get more words and verify the model knows them
            similar_words.append(
                pre_trained_embedding.most_similar(topic, topn*2))
            similar_words = lm_trigram.verify_words(similar_words)

        # generate lines
        output = lm_trigram.generate(length, similar_words)

        # render output/art:
        create_title(topic)
        # format poem
        print(make_lines(2, output))

    # close the program once all results have been printed
    sys.exit(0)


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python <topic> <poem_length> <model_type>")
        sys.exit(1)

    main()
