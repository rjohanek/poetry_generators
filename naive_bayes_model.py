import collections as c
import numpy as np
from pre_processing import pre_process
from joblib import dump

########################### NAIVE BAYES MODEL ######################################

# LanguageModel class represents a language model with the following capabilities:
# represent an n-gram model
# apply laplace smoothing (optionally)
# train on given sentences
# score given sentences (average probability) or files (average probability and standard deviance)
# generate one or any number of sentences


class LanguageModel:

    # the n-gram of this LanguageModel, default 2
    n = 2

    # if laplace smoothing will be applied, default False
    laplace = False

    # known vocabulary (n-grams) and their frequency of occurence based on training data
    vocab_counts = {}

    # the counts of all known tokens used in laplace smoothing
    token_counts = {}

    # the total number of n-grams this was trained on
    total_n_grams = 0

    # known vocabulary (n-grams) and their probability of occurence based on training data
    # this probability is used for scoring while the vocab_counts and total_n_grams are 
    # used to calculate probability in sentence generation
    vocab_probabilities = {}

    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
            n_gram (int): the n-gram order of the language model to create, must be 1 or greater
            is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        self.n = n_gram
        self.laplace = is_laplace_smoothing

    def train(self, training_data):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
            training_data (List of List of string): the training data to read

        Returns:
            None
            Saves learned data in the vocab_probabilities field
        """

        # create n-grams and tokens from list of list of tokens
        n_grams = []
        bigrams_backoff = []
        tokens = []
        for sentence in training_data:
            n_grams_sent = self.make_n_grams(sentence)

            # if this is an n > 2 model, also score the probabilities for bigrams
            if self.n > 2:
                n_grams_sent_backoff = self.make_n_grams(sentence, True)
                for bigram in n_grams_sent_backoff:
                    bigrams_backoff.append(bigram)

            for n_gram in n_grams_sent:
                n_grams.append(n_gram)
            for token in sentence:
                tokens.append(token)

        # calculate the total number of tokens trained on so that probabilities can be calculated
        self.total_n_grams = len(n_grams)

        # count the frequency of occurence for each n_gram and save this value on the Language Model
        self.vocab_counts = c.Counter(n_grams)
        self.token_counts = c.Counter(tokens)

        # calculate probabilities for each n_gram, applying laplace smoothing if specified
        # and save this value on the Language Model so it can be used in scoring
        self.vocab_probabilities = self.calculate_probabilities(
            self.vocab_counts, self.total_n_grams)

        # if this is an n>2 model, this will be used when there is an unknown n-gram context
        if len(bigrams_backoff) > 0:
            self.total_bigrams = len(bigrams_backoff)
            self.bigram_counts = c.Counter(bigrams_backoff)
            self.bigram_probabilities = self.calculate_probabilities(
                self.bigram_counts, self.total_bigrams)

    def make_n_grams(self, tokens, backoff=False):
        """Creates tuples of n length representing the n-grams in the given tokens
        Parameters:
          tokens (list of str): the list of individual tokens to be grouped into n-grams

        Returns:
          n_grams (list of tuples of str): the list of n-grams created from the given tokens
        """
        n_grams = []

        # unigrams are just tokens and will never need to use backoff
        if self.n == 1:
            return tokens

        n = self.n

        # generate bigrams
        if backoff:
            n = 2

        # generate n-grams where n is specified by the model
        for i in range(n - 1, len(tokens)):
            n_gram = [tokens[j] for j in range(i - (n - 1), i + 1)]
            n_grams.append(tuple(n_gram))
        return n_grams

    def calculate_probabilities(self, counts, num_n_grams):
        """Given the counts for each n_gram and the total number of n_grams calculate_probabilities calculates the 
             probability of occurence for each n_gram, using laplace smoothing if it is specified by the model
        Parameters:
            counts (Counter): mapping of n_grams to their frequency of occurence
            num_n_grams (int): total number of n_grams seen

        Returns:
            probabilities, a mapping of n_grams to their probability of occurence
        """

        if self.n == 1:
            return self.calculate_probabilities_for_unigram(counts, num_n_grams)

        else:
            return self.calculate_probabilities_for_n_gram(counts)

    def calculate_probabilities_for_unigram(self, counts, num_unigrams):
        """Given the counts for each unigram and the total number of unigrams this calculates the 
             probability of occurence for each unigram, using laplace smoothing if it is specified by the model
        Parameters:
            counts (Counter): mapping of unigrams to their frequency of occurence
            num_unigrams (int): total number of unigrams seen

        Returns:
            probabilities, a mapping of unigrams to their probability of occurence
        """

        probabilities = {}

        if self.laplace:
            for key in counts.keys():
                probabilities[key] = (
                    (counts[key] + 1) / (num_unigrams + len(counts.keys()) - 1))

        else:
            for key in counts.keys():
                probabilities[key] = (counts[key] / num_unigrams)

        return probabilities

    def calculate_probabilities_for_n_gram(self, counts):
        """Given the counts for all n_grams this calculates the probability of occurence for each n_gram,
             using laplace smoothing if it is specified by the model
        Parameters:
            counts (Counter): mapping of n_grams to their frequency of occurence

        Returns:
            probabilities, a mapping of n_grams to their probability of occurence
        """
        probabilities = {}

        # get the count of similar n-grams
        similar_n_grams = []
        for key in counts.keys():
            key_start = key[:-1]
            similar_n_grams.append(key_start)
        counts_of_similar_n_grams = c.Counter(similar_n_grams)

        # divide count of this n-gram by total counts of all similar n-grams
        for key in counts.keys():
            if self.laplace:
                probabilities[key] = (
                    counts[key] + 1) / (counts_of_similar_n_grams[key[:-1]] + len(self.token_counts.keys()))
            else:
                # avoid divide by zero error
                if (counts_of_similar_n_grams[key[:-1]] == 0):
                    probabilities[key] = 0
                else:
                    probabilities[key] = counts[key] / \
                        counts_of_similar_n_grams[key[:-1]]

        return probabilities

    def generate_sentence(self, init):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
            sentence: list of tokens, the generated sentence
        """

        sentence = []

        # add the necessary number of SENT_BEGIN
        # and randomly sample the probability distribution if this is a model unigram
        if self.n == 1:
            sentence.append(self.SENT_BEGIN)
            sentence.append(init)

            k = []
            v = []
            # don't include the sentence begin token
            total_grams = self.total_n_grams - \
                self.vocab_counts[self.SENT_BEGIN]
            for key in self.vocab_counts.keys():
                if key != self.SENT_BEGIN:
                    k.append(key)
                    v.append(self.vocab_counts[key] / total_grams)

            while sentence[-1] != self.SENT_END:
                sentence.append(np.random.choice(k, p=v))

        # randomly sample the probability distribution if this is a model an n-gram model, n > 1
        else:
            # add the necessary number of SENT_BEGIN
            for i in range(self.n - 1):
                sentence.append(self.SENT_BEGIN)
            sentence.append(init)

            # if this is the initial n-gram, backoff to a bigram model
            # generate the next most likely token given the last token in the n_gram so far
            # do this until you have an n-gram that is known.
            while len(sentence) < self.n + 1:
                ngram_sofar = sentence[-1]
                # find all possible next words and their probabilities
                possible_words_probs = self.get_possible_next_words_probs(
                    ngram_sofar, True)
                # sample the probability distribution (at 0: words, at 1: probabilities) and add to the sentence
                sentence.append(np.random.choice(
                    possible_words_probs[0], p=possible_words_probs[1]))

            while sentence[-1] != self.SENT_END:
                # get the prefix, n-gram so far
                end = len(sentence)
                start = end - (self.n - 1)
                ngram_sofar = sentence[start:end]

                # find all possible next words and their probabilities
                possible_words_probs = self.get_possible_next_words_probs(
                    ngram_sofar)

                # sample the probability distribution (at 0: words, at 1: probabilities) and add to the sentence
                sentence.append(np.random.choice(
                    possible_words_probs[0], p=possible_words_probs[1]))

        return sentence

    def get_possible_next_words_probs(self, ngram_sofar, backoff=False):
        ''' Given the n-gram so far, generate a list of possible next words and calculate their probabilities.

                Parameters: n-gram so far, the prefix

                Returns: list of the next possible words and a corresponding list of their probabilites
        '''
        words = []
        counts = []
        probs = []
        total_count = 0

        dictionary = self.vocab_counts.items()
        if backoff == True:
            dictionary = self.bigram_counts.items()

        # first pass
        # find all possible n-grams given the n-gram so far
        for key, count in dictionary:
            # identify the prefix of the n-gram key, all words in n-gram but last
            key_start = []
            if backoff == True:
                key_start = key[0]
            else:
                for w in key[:-1]:
                    key_start.append(w)

            # if the prefix of the n-gram key matches the ngram_sofar
            # then add the last word of the key to the list of possible words, and the count to counts
            if key_start == ngram_sofar:
                words.append(key[-1])
                counts.append(count)
                total_count += count

        # divide the count of a possible word with the count of all possible words
        for count in counts:
            probs.append(count / total_count)

        return (words, probs)

    def generate(self, n, topic_words):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
            n (int): the number of sentences to generate

        Returns:
            list: a list containing strings, one per generated sentence
        """
        text = []
        for i in range(n):
            text.append(self.generate_sentence(topic_words[i]))

        return text

    def verify_words(self, words):
        """Verifies that this model knows the given words, 
        if a word is not known, then remove it from the returned list

        Parameters: list of words to check if in vocab
        Returns: list of given words in vocab
        """
        known_words = list(self.token_counts.keys())
        given_known_words = []
        for w in words:
            if w in known_words:
                given_known_words.append(w)
        return given_known_words

######################## main method for training ##################################


def main():
    """
    Train and save the trigram naive bayes mode
    """
    training_data_tri = pre_process("PoetryFoundationData.csv", 3)
    print("done pre-processing data")

    my_clf = LanguageModel(3, False)
    my_clf.train(training_data_tri)
    print("done training trigram model")

    dump(my_clf, "my_clf.pkl")
    print("done saving trigram model")

    pass


if __name__ == '__main__':

    main()
