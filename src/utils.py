import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd 
import numpy as np
import torch
import tensorflow as tf
keras = tf.keras
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.pdf')
    plt.clf()

def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window : mid + window, :]

def getresults2(df, result):
    results2, df1, df2 = {}, df.sum(), df.mean()
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]
    for a in ['precision', 'recall']:
        results2[a] = df2[a]
    results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
    return results2

def simple_discretize_dataset(data: torch.Tensor, n_letters: int=4):
    """
    Discretizes time series columns into chains of symbols. Specially designed for Word2Vec approach.

    Parameters:
    -----------
    data: torch.Tensor, input dataset.
    n_letters: int, number of letters to discretize the time series.

    Returns:
    --------
    data: pd.Dataframe, contains all time series columns discretized into strings and combined in one column.
    """
    dis_data = np.empty((data.shape[0], data.shape[1]), dtype=object)
    for i in range(0, data.shape[1]):
        col = np.array(torch.index_select(data, 1, torch.tensor([i])))
        max_v = col.max()
        min_v = col.min()
        vfunc = np.vectorize(lambda x: str(chr(65+int((x-min_v)/(max_v-min_v+0.0001)*n_letters))))
        col = vfunc(col)
        dis_data[:, i] = col.squeeze()

    # Combine all the columns into one column
    data = pd.DataFrame(dis_data)
    data = data.apply(lambda x: ''.join(x.astype(str)), axis=1)
    data = pd.DataFrame(data, columns=['discretized_data'])

    return data

def generate_skipgrams(corpus, window_size, debug=False):
    """
    Generates skipgrams froma given corpus.

    Parameters:
    -----------
    corpus: string, the skipgrams will be generated with this string.
    window_size: int, size of the selected window to generate the skipgrams.
    debug: bool, variable to determine if after generate the skipgramss someof the will be
           printed in the terminal.

    Returns:
    --------
    skip_grams: list, generated skipgrams.
    word2id: dictionary, provides information about the relatioship between words and IDs.
    wids: 
    id2word: dictionary, provides informationabput the relationshipbetween words and IDs.
    """
    # Create and fit tokenizer with corpus
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(corpus)

    # Create dictionaries with relationship between Ids and words
    word2id = tokenizer.word_index
    id2word = {v: k for k, v in word2id.items()}

    vocab_size = len(word2id) 

    wids = [[word2id[w]
                for w in text.text_to_word_sequence(doc)] for doc in corpus]

    print('Vocabulary size:', vocab_size)
    print('Mos frequent words:', list(word2id.items())[-5:])

    # Generate skip-grams
    skip_grams = [
        skipgrams(wid, vocabulary_size=vocab_size, window_size=window_size) for wid in wids]

    # Show some skip-grams
    if debug:
        pairs, labels = skip_grams[0][0], skip_grams[0][1]
        for i in range(10):
            print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
                id2word[pairs[i][0]], pairs[i][0],
                id2word[pairs[i][1]], pairs[i][1],
                labels[i]))

    return skip_grams, word2id, wids, id2word