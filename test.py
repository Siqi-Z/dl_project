import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

public_directory = '/projects/training/bauh/COCO'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    total_step = len(data_loader)

    # List to score the BLEU scores
    bleu_scores = []

    for i, (images, captions, lengths) in enumerate(data_loader):
        
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        
        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        score = sentence_bleu(reference, candidate, args.bleu_weights)
        bleu_scores.append(score)

        # Print log info
        if i % args.log_step == 0:
            print('Finish [{}/{}], Current BLEU Score: {:.4f}'
                  .format(i, total_step, np.mean(bleu_scores)))

    np.save('test_results.npy', [bleu_scores, mean(bleu_scores)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/test_vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--image_dir', type=str, default='data/val_resized', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default=public_directory+'/annotations/captions_val2014.json', help='path for train annotation json file')
    
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--bleu_weights', type=float, default=(1, 0, 0, 0))
    args = parser.parse_args()
    print(args)
    main(args)