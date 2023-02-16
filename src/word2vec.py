import torch
import torch.nn as nn
import tensorflow as tf
keras = tf.kera # small issue with Pylance and Tensorflow
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
torch.manual_seed(1)

class SkipGramNS(nn.module):
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
        self.linear = nn.Linear(1, 1,)

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
        # shape: (batch size, embedding dimension)
        tgt_emb = self.w(tgt_word)
        # Look up the embeddings for the positive and negative context words.
        # shape: (batch size, nbr contexts, emb dim)
        ctx_emb = self.c(ctx_word)

        # Compute dot product
        score = torch.dot(tgt_emb, ctx_emb)
        score = nn.LogSigmoid(score)

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
    # create and fit tokenizer with corpus
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(corpus)

    # create dictionaries with relationship between Ids and words
    word2id = tokenizer.word_index
    id2word = {v: k for k, v in word2id.items()}

    vocab_size = len(word2id) 

    wids = [[word2id[w]
                for w in text.text_to_word_sequence(doc)] for doc in corpus]

    print('Vocabulary size:', vocab_size)
    print('Mos frequent words:', list(word2id.items())[-5:])

    # generate skip-grams
    skip_grams = [
        skipgrams(wid, vocabulary_size=vocab_size, window_size=window_size) for wid in wids]

    # show some skip-grams
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
    VOCAB_SIZE = 5
    EMBEDDING_DIM = 100
    EPOCHS = 3
    model = SkipGramNS(VOCAB_SIZE, EMBEDDING_DIM)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    c_epoch=0
    # training loop
    while c_epoch <= EPOCHS:
        loss = 0
        for i, elem in enumerate(skip_grams):
            # TODO 


