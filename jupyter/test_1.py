import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from Preprocess import load_captions
from data_loader import DataLoader
from data_loader import get_loader 
from Vocabulary import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

train_dir="train"
val_dir="dev"
encoder_path= "./models/encoder-10-200.ckpt"
decoder_path= "./models/decoder-10-200.ckpt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    threshold = 20
    captions_dict = load_captions(train_dir)
    vocab = Vocabulary(captions_dict, threshold)
    vocab_size=vocab.index
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([transforms.Resize((224, 224)), 
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5),
														 (0.5, 0.5, 0.5))
									])
    dataloader = DataLoader(val_dir, vocab, transform)
    imagenumbers, captiontotal, imagetotal= dataloader.gen_data()
                             
        # Build data loader
    data_loader = get_loader(imagenumbers, captiontotal, imagetotal, args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)


    # Build models
    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size, args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    
    # Build data loader

    total_step = len(data_loader)

    # List to score the BLEU scores
    bleu_scores = []

    for i, (images, captions, lengths) in enumerate(data_loader):
        
        # Set mini-batch dataset
        images = images.to(device)
        # captions = captions.to(device)
        
        # Generate an caption from the image
        feature = encoder(images)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.get_word(word_id)
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        score = sentence_bleu([captions], sentence, args.bleu_weights)
        bleu_scores.append(score)

        # Print log info
        if i % args.log_step == 0:
            print('Finish [{}/{}], Current BLEU Score: {:.4f}'
                  .format(i, total_step, np.mean(bleu_scores)))
            print(sentence)
            print(captions)

    np.save('test_results.npy', [bleu_scores, np.mean(bleu_scores)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--vocab_path', type=str, default=local_directory+'/vocab.pkl', help='path for vocabulary wrapper') # use train vocab
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    #parser.add_argument('--image_dir', type=str, default=local_directory+'/val_resized', help='directory for resized images') # FIXME: debugging, use train_images
   # parser.add_argument('--caption_path', type=str, default=public_directory+'/annotations/captions_val2014.json', help='path for train annotation json file')
    
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--bleu_weights', type=float, default=(0.25, 0.25, 0.25, 0.25))
    args = parser.parse_args()
    print(args)
    main(args)
