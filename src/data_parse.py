'''
data_parse.py

Reads and interprets data and metadata of
the 2018 VizWiz data set.

3/16/25
Lukas Zumwalt
'''

import json
# import numpy as np

# Directory for all images
IMG_DIR = '../data/images/'

# Directory for annotation files
ANN_DIR = '../data/annotations/'

TRAIN_ANNOTATION_PATH = f'{ANN_DIR}train.json'
VAL_ANNOTATION_PATH = f'{ANN_DIR}val.json'
TEST_ANNOTATION_PATH = f'{ANN_DIR}test.json'


if __name__ == '__main__':

    # Train
    with open(TRAIN_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # Validation
    with open(VAL_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Test
    with open(TEST_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print('Train set size:', len(train_data))
    print('Validation set size:', len(val_data))
    print('Test set size:', len(test_data))

    vq = train_data[0]

    # Try printing the entire set of annotation to see the structure
    print('First sample:', vq)

    # Unpack data: image
    image_name = vq['image']
    image_url = IMG_DIR + image_name
    print('Image URL:', image_url)

    # Unpack data: question
    question = vq['question']

    # Unpack data: labels
    answers = vq['answers']
    label = vq['answerable']
    print('Image name (file name):', image_name)
    print('Question:', question)
    print('Answer index 0:', answers[0])
    print('Answerability label:', label)
