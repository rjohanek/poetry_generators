
import pandas as pd
import re

START = "<s>"
END = "</s>"
line_delim = "\n"


def tokenize(sentences):
    """break up the given sentences into tokens within the sentence and normalize the text"""
    sentences_tokenized = []
    # lowercase the text, remove unwanted punctuation, tokenize each sentence
    # and add the list of tokens to the list of tokenized sentences
    for sent in sentences:
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
    data = pd.read_csv(filePath)
    poems = []
    for row in data.iterrows():
        # row[0] is the id
        # row[1] is the info: index at [0], title at [1], poem at [2], poet at [3], and tags at [4]
        poems.append(row[1][2])

    poems_joined_lines = line_delim.join(poems)

    # example of line: Dog bone, stapler,
    lines = tokenize(poems_joined_lines.split('\n'))
    lines = [x for x in lines if len(x) > 2]

    # if n is greater than two add correct number of sentence start tokens
    for i in range(n-2):
        lines = pre_pend(lines)

    return lines
