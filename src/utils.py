import matplotlib.pyplot as plt
import os
import random
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

def prepare_discretized_data(train_data, test_data, model, debug=False):
    # Discretize dataset
    train_test = torch.cat((train_data, test_data), 0)
    full_data, train_data, test_data = simple_discretize_dataset(train_test, train_length=train_data.shape[0],  n_letters=model.n_letters)
    train_corpus = dataframe_to_corpus(train_data)

    # Full data is required to obtain vocabulary size
    full_data = dataframe_to_corpus(full_data)
    vocab_size = obtain_vocab_size(full_data)

    # Generate skipgrams for training
    skip_grams, word2id, wids, id2word = generate_skipgrams(train_corpus, model.n_window, debug)

    # Generate word pair for POT evaluation
    train_data = [
        torch.tensor(wids[0][1:]),
        torch.tensor(wids[0][:-1])]

    # Generate test data by delaying the test time series by one unit to create the word pairs
    test_corpus = dataframe_to_corpus(test_data)
    test_wids = [[]]

    # Some test words do not seem to have appeared in train
    for w in text.text_to_word_sequence(test_corpus[0]):
        if not w in word2id:
            word2id[w] = len(word2id) + 1
        test_wids[0].append(word2id[w])
    
    # The context in the testing set is the previous point
    test_data = [
        torch.tensor(test_wids[0][1:]),
        torch.tensor(test_wids[0][:-1])]

    return skip_grams, train_data, test_data, vocab_size

def prepare_discretized_data_keras(train_data, test_data, model, debug=False):
    train_test = torch.cat((train_data, test_data), 0).detach().numpy()
    full_data, train_data, test_data = simple_discretize_dataset_np(train_test, train_length=train_data.shape[0],  n_letters=model.n_letters)
    train_corpus = dataframe_to_corpus(train_data)
    
    # Full data is required to obtain vocabulary size
    full_data = dataframe_to_corpus(full_data)
    vocab_size = obtain_vocab_size(full_data)

    # Generate skipgrams for training
    skip_grams, word2id, wids, id2word = generate_skipgrams(train_corpus, model.n_window, debug)
    
    # Generate word pair for POT evaluation
    train_data = [
        np.array(wids[0][1:], dtype='int32'),
        np.array(wids[0][:-1], dtype='int32')]

    # Generate word pair for POT evaluation
    train_data = [
        np.array(wids[0][1:], dtype='int32'),
        np.array(wids[0][:-1], dtype='int32')]
    
    # Generate test data by delaying the test time series by one unit to create the word pairs
    test_corpus = dataframe_to_corpus(test_data)
    test_wids = [[]]
    
    # Some test words do not seem to have appeared in train
    for w in text.text_to_word_sequence(test_corpus[0]):
        if not w in word2id:
            word2id[w] = len(word2id) + 1
        test_wids[0].append(word2id[w])
    
    # The context in the testing set is the previous point
    test_data = [
        np.array(test_wids[0][1:], dtype='int32'),
        np.array(test_wids[0][:-1], dtype='int32')]
    
    return skip_grams, train_data, test_data, vocab_size

def simple_discretize_dataset(data, train_length, n_letters=5):
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
    train_data = data.iloc[:train_length]
    test_data = data.iloc[train_length:]

    return data, train_data, test_data

def simple_discretize_dataset_np(data, train_length, n_letters=5):
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
        col = np.array(data[:, i])
        max_v = col.max()
        min_v = col.min()
        vfunc = np.vectorize(lambda x: str(chr(65+int((x-min_v)/(max_v-min_v+0.0001)*n_letters))))
        col = vfunc(col)
        dis_data[:, i] = col.squeeze()

    # Combine all the columns into one column
    data = pd.DataFrame(dis_data)
    data = data.apply(lambda x: ''.join(x.astype(str)), axis=1)
    train_data = data.iloc[:train_length]
    test_data = data.iloc[train_length:]

    return data, train_data, test_data

def dataframe_to_corpus(data: pd.DataFrame):
        # Convert Dataframe to a large string containing all words
        corpus = []
        smd_text = ""  # Empty string
        # Fill corpus
        for i in data:
            smd_text += i
            smd_text += " "

        corpus.append(smd_text)

        return corpus

def obtain_vocab_size(corpus, debug=False):
    # Create and fit tokenizer with a given corpus
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(corpus)
    if debug:
        print('Vocabulary size:', len(tokenizer.word_index))
    return len(tokenizer.word_index)

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
    print('Most frequent words:', list(word2id.items())[-5:])

    #import csv
    #with open('tokenizer_machine-1-1.csv','w') as f:
    #    w = csv.writer(f)
    #    w.writerows(tokenizer.word_counts.items())

    # Generate skip-grams
    skip_grams = [
        skipgrams(wid, vocabulary_size=vocab_size, window_size=window_size) for wid in wids]

    # Show some skip-grams
    if debug:
        pairs, labels = skip_grams[0][0], skip_grams[0][1]
        for i in range(5):
            print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
                id2word[pairs[i][0]], pairs[i][0],
                id2word[pairs[i][1]], pairs[i][1],
                labels[i]))

    return skip_grams, word2id, wids, id2word

def estimate_perplexity(y_pred):
    # Since we are working with pairs of words, this is a bigram
    # High probability will be translated into less perplex
    return (1.0/y_pred)

def downsampling(data, labels=None, factor=10):
    print("[+] DEBUG: factor = " + str(factor))
    # Downsamples data with a given factor
    data = data.detach().numpy() # Convert from Pytorch tensor to Numpy array
    new_data = []
    # Iterate over all columns if the time series is multivariate
    for col in range(0, data.shape[1]):
        i = 0
        new_data_col = []
        while i < data.shape[0]:
            if i+factor > data.shape[0]:
                # Special case for last term
                next = data.shape[0]
            else:
                next = i+factor

            new_data_col.append([np.mean(data[i:next][col])])
            i = next
        # Return Numpy arrays to Pytorch format
        new_data.append(np.array(new_data_col).squeeze())
    
    if labels is not None:
        j = 0
        new_labels = []
        while j < data.shape[0]:
            if j+factor > data.shape[0]:
                # Special case for last term
                next = data.shape[0]
            else:
                next = j+factor  

            if np.where(labels[j:next] == 1)[0].size == 0:
                # If the array is empty there aren't anomaly points in the interval
                # so the downsampled data will be normal
                new_labels.append([0.])
            else:
                # If the array is not empty there are anomlay points in the interval
                # so the downsampled data will be an anomaly
                new_labels.append([1.])
            j = next    
        new_labels = np.array(new_labels)
        
    # Return Numpy arrays to Pytorch format
    new_data = torch.from_numpy(np.array(new_data)).T
    print("[+] DEBUG: New shape = " + str(new_data.shape[0]))
    
    return new_data, new_labels


def skipgrams_k(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None, k=4):
    """
    Modification of skipgramns implementation in Keras library.
    The pairs are generated with a delay of 'k' time units.

    Parameters:
    -----------
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        shuffle: Whether to shuffle the word couples before returning them.
        categorical: bool. if False, labels will be
            integers (eg. `[0, 1, 1 .. ]`),
            if `True`, labels will be categorical, e.g.
            `[[1,0],[0,1],[0,1] .. ]`.
        sampling_table: 1D array of size `vocabulary_size` where the entry i
            encodes the probability to sample a word of rank i.
        seed: Random seed.

    Returns:
    --------
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.
    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue
        if i<k:
            # Only for initialization in the time series
            continue
        wj = sequence[i-k]
        couples.append([wi, wj])
        if categorical:
            labels.append([0, 1])
        else:
            labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [
            [words[i % len(words)], random.randint(1, vocabulary_size - 1)]
            for i in range(num_negative_samples)
        ]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels

