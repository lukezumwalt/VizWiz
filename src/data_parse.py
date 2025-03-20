'''
data_parse.py

Reads and interprets data and metadata of
the 2018 VizWiz data set.

Lukas Zumwalt
3/16/25
'''

import json
from collections import Counter
from skimage import io
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import BertTokenizer

# Directory for all images
IMG_DIR = '../data/images/'

TRAIN_IMG_PATH = IMG_DIR + 'train/'
VAL_IMG_PATH = IMG_DIR + 'val/'
TEST_IMG_PATH = IMG_DIR + 'test/'

# Directory for annotation files
ANN_DIR = '../data/annotations/'

TRAIN_ANNOTATION_PATH = f'{ANN_DIR}train.json'
VAL_ANNOTATION_PATH = f'{ANN_DIR}val.json'
TEST_ANNOTATION_PATH = f'{ANN_DIR}test.json'


class DataStruct():
    '''
    Dataset class designed to hold pointers
    for respective data in accessible member form.
    '''

    def __init__(self,
                 images,
                 annotations,
                 txform=None,
                 tokenizer=None,
                 subset=0,
                 token_max_len=32):

        self.image_path = images
        self.annotation_path = annotations
        self.txform = txform
        self.tokenizer = tokenizer
        self.token_max_len = token_max_len  # max length limit for tokens

        # Load annotations and filter for a subset if provided
        with open(self.annotation_path, 'r', encoding='utf-8') as fn:
            self.annotations = json.load(fn)
        if subset != 0:
            self.subset = self.annotations[:subset]

        self.chosen_answers = []
        for sample in self.annotations:
            answers = [entry['answer'] for entry in sample ['answers']]
            answer_counts = Counter(answers)
            top_answer,_ = answer_counts.most_common(1)[0]
            self.chosen_answers.append(top_answer)
        answer_counts = Counter(self.chosen_answers)
        self.top_answers = answer_counts.most_common(100)
        print(self.top_answers)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,idx):
        annotation = self.annotations[idx]
        image_name = annotation['image']
        question = annotation['question']
        labels = annotation['answers']
        image = self.image_path[idx] + image_name

        if self.txform:
            image = self.txform(image)

        # Tokenize text if tokenizer is provided
        if self.tokenizer:
            encoding = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=self.token_max_len,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)      # shape: [max_length]
            attention_mask = encoding['attention_mask'].squeeze(0)  # shape: [max_length]
        else:
            input_ids = None
            attention_mask = None

        return image_name, question, labels, input_ids, attention_mask

    def show(self, idx, rich=False):
        '''
        Method to depict an indexed image and more metadata.
        '''
        sample = self.annotations[idx]
        if rich:
            print('Sample:\n', sample)
        sample_image_name = sample['image']
        sample_image_path = self.image_path + sample_image_name
        sample_question = sample['question']
        sample_answers = sample['answers']
        sample_label = sample['answerable']
        # Printing
        print('Image name (file name):', sample_image_name)
        print('Question:', sample_question)
        if rich:
            print('Answers:')
            for _,n in enumerate(sample_answers):
                print(n)
        else:
            print('Answer index 0:', sample_answers[0])
        print('Answerability label:', sample_label)
        # View image sample
        self.visualize_image(sample_image_path)

    # Plot image from the image url
    def visualize_image(self, image_path):
        '''
        Call to read image path and depict.
        '''
        image = io.imread(image_path)
        print(image_path)
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == '__main__':

    # Training
    with open(TRAIN_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # Validation
    with open(VAL_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Testing
    with open(TEST_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Size reporting
    print('--------------------------------')
    print('Train set size:', len(train_data))
    print('Validation set size:', len(val_data))
    print('Test set size:', len(test_data))

    #################################
    # Struct Handling               #
    #################################
    
    # Pre-Processing Objects
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Declare a training data struct
    training_data = DataStruct(TRAIN_IMG_PATH, TRAIN_ANNOTATION_PATH, transform, tokenizer)

    # Visualize top training answers
    bar_labels, bar_counts = zip(*training_data.top_answers)

    # Create a bar chart
    plt.figure(figsize=(20,10))  # Increase figure size for better readability
    plt.bar(bar_labels, bar_counts)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Bar Chart of Labels and Counts')
    plt.xticks(rotation=90)  # Rotate x labels for readability
    plt.tight_layout()       # Adjust layout to ensure everything fits
    plt.show()

    # Depict an indexed image
    training_data.show(10,True)

    # print(training_data[5])
