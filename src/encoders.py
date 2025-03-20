'''
encoders.py

Provides pre-process encoding to images and text
provided for the VizWiz VQA data set.

3/19/25
Lukas Zumwalt
'''
from torch import nn
from transformers import BertModel

class ImageEncoder(nn.Module):
    '''
    Image Encoder, small CNN / pretrained backbone
    '''
    def __init__(self, pretrained_backbone=None, embed_dim=256):
        super().__init__()
        # For simplicity, define a small CNN or load a pretrained model
        if pretrained_backbone is not None:
            self.cnn = pretrained_backbone
            # freeze or partially freeze layers if desired
            # for param in self.cnn.parameters():
            #     param.requires_grad = False
            # then add a linear layer to map to embed_dim
            self.fc = nn.Linear(1000, embed_dim)  # if the backbone outputs 1000-d
        else:
            # A trivial CNN definition (placeholder)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, 3, H, W) image
        """
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        embeddings = self.fc(features)
        return embeddings  # shape: (batch_size, embed_dim)

class TextEncoder(nn.Module):
    '''
    BERT Text Encoder
    '''
    def __init__(self, pretrained_model_name='bert-base-uncased', embed_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        # Map the [CLS] hidden state to desired embed_dim
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token hidden state
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        embeddings = self.fc(cls_hidden)                # shape: (batch_size, embed_dim)
        return embeddings