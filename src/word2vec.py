import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
keras = tf.keras # small issue with Pylance and Tensorflow
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from tqdm import tqdm
torch.manual_seed(1)

class SkipGramNS(nn.Module):
    """
    Skip-Gram Negative Sampling implementation with Pytorch library.
    DO NOT USE with experiments, only testing purposes.
    Only use with the inner main in this file.
    """
    def __init__(self, vocab_size, embed_size) -> None:
        super(SkipGramNS, self).__init__()
        self.name = 'SkipGramNS'
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # Target word embeddings
        self.w = nn.Embedding(vocab_size, embed_size, sparse=True)
        # Context embeddings
        self.c = nn.Embedding(vocab_size, embed_size, sparse=True)
        # Linear output layer
        # self.linear = nn.Linear(1, 1,)

        # Initialize embeddings
        self.init_emb()
    
    def init_emb(self):
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.c.weight)

    def forward(self, tgt_word, ctx_word):
        """
        tgt_word: target word ID
        ctx_word: context word ID
        """
        # Look up the embeddings for the target words
        tgt_emb = self.w(tgt_word)
        # Look up the embeddings for the context words
        ctx_emb = self.c(ctx_word)

        # Compute dot product
        score = torch.dot(tgt_emb, ctx_emb)
        return torch.sigmoid(score)

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

def main():
    print('Running main: testing SkipGramNS')
    corpus = ['A B B C A D D E D A B B B C C C A C C B D E A A A']
    skip_grams, word2id, wids, id2word = generate_skipgrams(corpus, 3, debug=True)
    VOCAB_SIZE = 5 + 1
    EMBEDDING_DIM = 100
    EPOCHS = 3
    model = SkipGramNS(VOCAB_SIZE, EMBEDDING_DIM)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.05)
    loss = nn.MSELoss()
    epoch = 0
    # training loop
    for e in tqdm(list(range(epoch+1, epoch+EPOCHS+1))):
        print('[+] Epoch nª {}'.format(e))
        for i, pair in enumerate(skip_grams[0][0]):
            # Prepare data
            target_word = torch.tensor(pair[0])
            context_word = torch.tensor(pair[1])
            Y = torch.tensor(skip_grams[0][1][i])

            # Compute the output from the model
            # That is, the dot products between target embeddings
            # and context embeddings.
            scores = model(target_word, context_word)

            # Compute loss
            l = loss(scores, Y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # Debug info
            if i % 100 == 0:
                print(' [+] Processed {} pairs'.format(i))

if __name__ == "__main__":
    main()
