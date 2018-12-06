import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import os
import pickle
import numpy as np
import nltk
from PIL import Image


class DataLoader():
	def __init__(self, dir_path, vocab, transform):
		self.images = None
		self.captions_dict = None
		# self.data = None
		self.vocab = vocab
		self.transform = transform
		self.load_captions(dir_path)
		self.load_images(dir_path)


	def load_captions(self, captions_dir):
		caption_file = os.path.join(captions_dir, 'captions.txt')
		captions_dict = {}
		with open(caption_file) as f:
			for line in f:
				cur_dict = json.loads(line)
				for k, v in cur_dict.items():
					captions_dict[k] = v
		self.captions_dict = captions_dict
	
	def load_images(self, images_dir):
		files = os.listdir(images_dir)
		images = {}
		for cur_file in files:
			ext = cur_file.split('.')[1]
			if ext == 'jpg':
				images[cur_file] = self.transform(Image.open(os.path.join(images_dir, cur_file)).convert('RGB'))
		self.images = images
	
	def caption2ids(self, caption):
		vocab = self.vocab
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		vec = []
		vec.append(vocab.get_id('<start>'))
		vec.extend([vocab.get_id(word) for word in tokens])
		vec.append(vocab.get_id('<end>'))
		return vec
	
	def gen_data(self):
		imagenumbers = []
		captions = []
		for image_id, cur_captions in self.captions_dict.items():
			num_captions = len(cur_captions)
			imagenumbers.extend([image_id] * num_captions)
			for caption in cur_captions:
				captions.append(self.caption2ids(caption))
		# self.data = images, captions
		#data = images, captions
		return imagenumbers, captions, self.images

class FlickrDataset(data.Dataset):
    """Flickr Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, images, captions,imagenumbers):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.imagenumbers=imagenumbers
        self.images=images
        self.captions=captions

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        imagenumber=self.imagenumbers[index]
        image=self.images[imagenumber]

        # Convert caption (string) to word ids.
        caption = self.captions[index]
        target = torch.Tensor(caption)
        
        return image, target

    def __len__(self):
        return len(self.imagenumbers)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(imagenumbers, captions, images, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    flickr = FlickrDataset(images=images,
                       captions=captions,
                       imagenumbers=imagenumbers)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=flickr, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
