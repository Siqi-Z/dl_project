import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO

public_directory = '/projects/training/bauh/Flicker8k_Dataset'
local_direcotry = '/u/training/tra.../scratch/data'

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    train_vocab = build_vocab(json=args.train_caption_path, threshold=args.threshold)
    test_vocab = build_vocab(json=args.test_caption_path, threshold=args.threshold)

    train_vocab_path = args.train_vocab_path
    with open(train_vocab_path, 'wb') as f:
        pickle.dump(train_vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(train_vocab_path))

    test_vocab_path = args.test_vocab_path
    with open(test_vocab_path, 'wb') as f:
        pickle.dump(test_vocab_path, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(test_vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_caption_path', type=str, 
                        default=public_directory+'/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--test_caption_path', type=str, 
                        default=public_directory+'/annotations/captions_val2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--train_vocab_path', type=str, default=local_directory+'/train_vocab.pkl', 
                        help='path for saving train vocabulary wrapper')
    parser.add_argument('--test_vocab_path', type=str, default=local_directory+'/test_vocab.pkl', 
                        help='path for saving test vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
