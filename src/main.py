#!/usr/bin/env python3

import os
import json
import pickle
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io

from transformers import BertTokenizer, BertModel

############################################################
# 1. DataStruct Class (Local Files + OOP Improvements)
############################################################
class DataStruct(Dataset):
    """
    A Dataset-like class to hold and process the VizWiz data in an
    object-oriented manner. We fix some of the weaknesses by:

    1) Reading image paths and opening them in __getitem__ so that
       transforms are actually applied to an image tensor (not a string).
    2) Providing top-answer logic for classification tasks.
    3) Storing 'answerable' for binary classification.
    4) Storing question text, tokenizing with BERT.
    5) Exposing a show() method for visualization and debugging.
    """

    def __init__(
        self,
        image_dir,            # e.g. "../data/images/train/"
        annotation_path,      # e.g. "../data/annotations/train.json"
        txform=None,
        tokenizer=None,
        subset=0,
        token_max_len=32,
        top_n=100,
        build_top_answers=True
    ):
        """
        Args:
            image_dir: Directory containing the images (local).
            annotation_path: Path to the annotation .json file.
            txform: torchvision transforms for images.
            tokenizer: BERT tokenizer for text.
            subset: if non-zero, we only load a subset of the data for debugging.
            token_max_len: maximum length for text tokens.
            top_n: number of top answers to keep for classification (Challenge 2).
            build_top_answers: if True, we compute the top answers from the data.
        """
        super().__init__()
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.txform = txform
        self.tokenizer = tokenizer
        self.token_max_len = token_max_len
        self.top_n = top_n

        # Load annotations
        with open(self.annotation_path, 'r', encoding='utf-8') as fn:
            self.annotations = json.load(fn)

        # If subset is specified, keep only that many
        if subset != 0:
            self.annotations = self.annotations[:subset]

        # Build top answers if requested
        # (Challenge 2 approach: pick the single "most common" from each sample’s 10 answers)
        self.top_answers = None
        self.answer_to_idx = {}
        self.idx_to_answer = {}

        if build_top_answers:
            chosen_answers = []
            for sample in self.annotations:
                # from "answers": pick the most common
                answers = [entry['answer'] for entry in sample['answers']]
                counts = Counter(answers)
                top_answer, _ = counts.most_common(1)[0]
                chosen_answers.append(top_answer)

            # Count frequencies across all samples
            overall_counts = Counter(chosen_answers)
            # Keep top N
            self.top_answers = overall_counts.most_common(self.top_n)
            # Build dictionaries
            for i, (ans, _) in enumerate(self.top_answers):
                self.answer_to_idx[ans] = i
            # We'll treat anything outside top_n as 'other_categories'
            # which we store at index top_n
            self.answer_to_idx["<OTHER>"] = self.top_n

            # Also store the reverse map
            for k, v in self.answer_to_idx.items():
                self.idx_to_answer[v] = k

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns:
            image_tensor: the transformed image
            question_encoding: dict of {'input_ids', 'attention_mask'} for BERT
            top_answer_idx: integer for the "most common" label among top_n (or top_n for 'other')
            answerable_label: int or float for the binary classification
            raw_meta: dictionary with raw data if needed (image name, question, etc.)
        """
        ann = self.annotations[idx]
        image_name = ann['image']           # e.g. "VizWiz_train_00000000.jpg"
        question_text = ann['question']     # e.g. "What is this?"
        answerable_label = ann.get('answerable', 0)  # 1 or 0

        # Possibly pick the single "most common" from the 10 answers
        answers = ann.get('answers', [])

        if answers:
            counts = Counter([a['answer'] for a in answers])
            top_answer, _ = counts.most_common(1)[0]
            if self.top_answers is not None:
                top_answer_idx = self.answer_to_idx.get(top_answer, self.answer_to_idx["<OTHER>"])
            else:
                top_answer_idx = -1
        else:
            top_answer_idx = -1


        # Load image from local directory
        img_path = os.path.join(self.image_dir, image_name)
        with Image.open(img_path).convert("RGB") as pil_img:
            if self.txform:
                image_tensor = self.txform(pil_img)
            else:
                # default to a simple ToTensor
                image_tensor = transforms.ToTensor()(pil_img)

        # Tokenize question
        if self.tokenizer:
            encoding = self.tokenizer(
                question_text,
                padding='max_length',
                truncation=True,
                max_length=self.token_max_len,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)      # shape: [token_max_len]
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            # dummy if no tokenizer
            input_ids = torch.zeros(self.token_max_len, dtype=torch.long)
            attention_mask = torch.zeros(self.token_max_len, dtype=torch.long)

        # Return everything needed
        # We can return a dictionary or a tuple
        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "top_answer_idx": top_answer_idx,
            "answerable": float(answerable_label),
            "raw_meta": {
                "image_name": image_name,
                "question": question_text,
                "answers": answers
            }
        }

    def show(self, idx, rich=False):
        """
        Visualize the image and metadata at index `idx`.
        """
        ann = self.annotations[idx]
        image_name = ann['image']
        question = ann['question']
        answers = ann['answers']
        label = ann.get('answerable', 0)

        print('Image name (file name):', image_name)
        print('Question:', question)
        if rich:
            print('Answers:')
            for a in answers:
                print(a)
        else:
            if len(answers) > 0:
                print('Answer index 0:', answers[0])
        print('Answerability label:', label)

        # Show the image
        sample_image_path = os.path.join(self.image_dir, image_name)
        self.visualize_image(sample_image_path)

    @staticmethod
    def visualize_image(image_path):
        """
        Load and display image via skimage + matplotlib
        """
        image = io.imread(image_path)
        print(image_path)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

############################################################
# 2. Example Model for Challenge 1 & 2
############################################################
class SimpleVQAModel(nn.Module):
    """
    Example multi-modal model:
    - Pretrained BERT for text (optionally frozen).
    - A small CNN for images.
    - Fusion: elementwise multiplication or a small feed-forward
    - Output: flexible heads for either binary classification or multi-class.
    """
    def __init__(self, num_answers=101, freeze_bert=True, hidden_dim=256):
        """
        num_answers = top_n + 1 (for 'other') if using classification approach
        freeze_bert = whether to freeze BERT params
        hidden_dim = dimension for fused representation
        """
        super().__init__()

        # BERT for question
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_hidden_dim = self.bert.config.hidden_size

        # Map BERT [CLS] to a smaller embedding
        self.text_proj = nn.Linear(self.bert_hidden_dim, hidden_dim)

        # Simple CNN for images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
        )
        # Flatten size is unknown until we do a test pass, so let's guess or do a small test
        self.cnn_out_dim = 32*7*7
        self.img_proj = nn.Linear(self.cnn_out_dim, hidden_dim)

        # Fusion
        # Let's do a simple elementwise multiplication
        # followed by a feed-forward
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        # Heads:
        # 1) Binary classification (for answerable)
        self.binary_head = nn.Linear(hidden_dim, 1)

        # 2) Multi-class classification (for top_n answers + "other")
        self.answer_head = nn.Linear(hidden_dim, num_answers)

    def forward(self, images, input_ids, attention_mask):
        """
        Return:
          fused: the fused representation
          binary_logits: shape (B,1)
          answer_logits: shape (B,num_answers)
        """
        # Text
        with torch.set_grad_enabled(not self.bert.parameters().__next__().requires_grad):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = bert_out.last_hidden_state[:, 0, :]  # (B, 768) for base uncased
        text_emb = self.text_proj(cls_hidden)             # (B, hidden_dim)

        # Image
        img_features = self.cnn(images)                   # (B, 32*7*7)
        img_emb = self.img_proj(img_features)             # (B, hidden_dim)

        # Fusion
        fused = img_emb * text_emb
        fused = self.fusion_fc(fused)                     # (B, hidden_dim)

        # Heads
        binary_logits = self.binary_head(fused)           # (B, 1)
        answer_logits = self.answer_head(fused)           # (B, num_answers)

        return fused, binary_logits, answer_logits

############################################################
# 3. Training / Evaluation Routines
############################################################

def train_one_epoch(model, dataloader, optimizer, device, ce_loss, bce_loss):
    """
    We simultaneously train for:
     - binary classification (answerable)
     - multi-class classification (top answer)
    Just as an example. If you only want one, comment out accordingly.
    """
    model.train()
    total_bin_loss = 0
    total_ans_loss = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        top_answer_idx = batch["top_answer_idx"].to(device)
        answerable = batch["answerable"].to(device).float()

        optimizer.zero_grad()
        _, binary_logits, answer_logits = model(images, input_ids, attention_mask)

        # Binary classification loss
        bin_loss = bce_loss(binary_logits.squeeze(1), answerable)

        # Multi-class classification loss
        ans_loss = ce_loss(answer_logits, top_answer_idx.long())

        # Combine them (just sum for demonstration)
        loss = bin_loss + ans_loss
        loss.backward()
        optimizer.step()

        total_bin_loss += bin_loss.item()
        total_ans_loss += ans_loss.item()

    avg_bin_loss = total_bin_loss / len(dataloader)
    avg_ans_loss = total_ans_loss / len(dataloader)
    return avg_bin_loss, avg_ans_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate for both tasks: binary classification + multi-class top answer
    Return a simple accuracy for each (not the official VizWiz metric).
    """
    model.eval()
    total_bin_correct = 0
    total_bin_count = 0

    total_ans_correct = 0
    total_ans_count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        top_answer_idx = batch["top_answer_idx"].to(device)
        answerable = batch["answerable"].to(device).float()

        _, binary_logits, answer_logits = model(images, input_ids, attention_mask)
        # Binary
        preds_bin = (torch.sigmoid(binary_logits) > 0.5).float()
        correct_bin = (preds_bin.squeeze(1) == answerable).sum().item()
        total_bin_correct += correct_bin
        total_bin_count += answerable.size(0)

        # Multi-class
        preds_ans = answer_logits.argmax(dim=1)
        correct_ans = (preds_ans == top_answer_idx).sum().item()
        total_ans_correct += correct_ans
        total_ans_count += top_answer_idx.size(0)

    bin_acc = total_bin_correct / total_bin_count if total_bin_count > 0 else 0
    ans_acc = total_ans_correct / total_ans_count if total_ans_count > 0 else 0
    return bin_acc, ans_acc

############################################################
# 4. Metric for Answerable (Binary) and Official VQA Score
############################################################

def vqa_official_score(pred_answer, human_answers):
    """
    Given a predicted answer (string) and a list of 10 human answers,
    the VizWiz official scoring is:
       # humans that provided that answer / 3, capped at 1
    """
    pred_answer = pred_answer.lower()
    human_list = [h['answer'].lower() for h in human_answers]
    count = sum(1 for x in human_list if x == pred_answer)
    return min(count / 3, 1.0)

def evaluate_vqa_official(model, dataloader, device, idx_to_answer):
    """
    Evaluate using the official VizWiz metric for the multi-class
    (top answer) portion. We'll ignore the binary portion here.
    """
    model.eval()
    total_score = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # For official scoring, we need the raw answers
            raw_batch = batch["raw_meta"]  # list of dicts
            top_answer_idx = batch["top_answer_idx"].to(device)

            _, _, answer_logits = model(images, input_ids, attention_mask)
            preds_ans = answer_logits.argmax(dim=1).cpu().numpy()

            # For each sample, compute official metric
            for i, pred_idx in enumerate(preds_ans):
                pred_str = idx_to_answer.get(pred_idx, "<OTHER>")
                # human answers
                human_answers = raw_batch[i]["answers"]
                score = vqa_official_score(pred_str, human_answers)
                total_score += score
                total_count += 1

    return total_score / total_count if total_count > 0 else 0.0

def custom_collate(batch):
    # For each key in the sample dict, if the key is "raw_meta",
    # then simply return a list of those dictionaries.
    collated = {}
    for key in batch[0]:
        if key == "raw_meta":
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = torch.utils.data.default_collate([sample[key] for sample in batch])
    return collated


############################################################
# 5. Generate Submission
############################################################
def generate_submission(model, dataloader, device, idx_to_answer, submission_file="submission.json"):
    """
    For Challenge 2 submission format:
       results = [
         {
           "image": "VizWiz_test_00000000.jpg",
           "answer": "some predicted answer"
         },
         ...
       ]
    We'll also create a .pkl for binary classification (Challenge 1).
    """
    model.eval()
    results = []
    binary_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            raw_batch = batch["raw_meta"]  # to retrieve image name
            _, binary_logits, answer_logits = model(images, input_ids, attention_mask)

            # Binary
            preds_bin = (torch.sigmoid(binary_logits) > 0.5).long().cpu().tolist()
            binary_preds.extend(preds_bin)

            # Multi-class
            preds_ans = answer_logits.argmax(dim=1).cpu().tolist()
            for i, pred_idx in enumerate(preds_ans):
                image_name = raw_batch[i]["image_name"]
                pred_str = idx_to_answer.get(pred_idx, "<OTHER>")
                results.append({
                    "image": image_name,
                    "answer": pred_str
                })

    # Save the multi-class answers (Challenge 2)
    with open(submission_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved Challenge 2 submission to {submission_file}.")

    # Save the binary predictions (Challenge 1)
    with open("challenge1_binary_preds.pkl", "wb") as f:
        pickle.dump(binary_preds, f)
    print("Saved Challenge 1 predictions to challenge1_binary_preds.pkl.")

############################################################
# 6. MAIN
############################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Example local paths (adjust as needed)
    # E.g.:
    # train_images_path = "data/Images/train/"
    # val_images_path   = "data/Images/val/"
    # test_images_path  = "data/Images/test/"
    # train_annotation_path = "data/Annotations/train.json"
    # val_annotation_path   = "data/Annotations/val.json"
    # test_annotation_path  = "data/Annotations/test.json"

    # For demonstration, define some placeholders:
    train_images_path = "../data/Images/train/"
    val_images_path   = "../data/Images/val/"
    test_images_path  = "../data/Images/test/"
    train_annotation_path = "../data/Annotations/train.json"
    val_annotation_path   = "../data/Annotations/val.json"
    test_annotation_path  = "../data/Annotations/test.json"

    # Basic transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create DataStruct objects
    # For speed, let's do subset=100, but in real usage, set subset=0 or None
    train_data = DataStruct(
        image_dir=train_images_path,
        annotation_path=train_annotation_path,
        txform=transform,
        tokenizer=tokenizer,
        subset=500,        # load 1000 samples for demonstration
        token_max_len=24,
        top_n=100,
        build_top_answers=True
    )
    val_data = DataStruct(
        image_dir=val_images_path,
        annotation_path=val_annotation_path,
        txform=transform,
        tokenizer=tokenizer,
        subset=200,         # smaller subset for val
        token_max_len=24,
        top_n=100,
        build_top_answers=False  # We don't rebuild top answers in val
    )
    # We share the top answers from train_data
    val_data.top_answers = train_data.top_answers
    val_data.answer_to_idx = train_data.answer_to_idx
    val_data.idx_to_answer = train_data.idx_to_answer

    test_data = DataStruct(
        image_dir=test_images_path,
        annotation_path=test_annotation_path,
        txform=transform,
        tokenizer=tokenizer,
        subset=100,         # for demonstration
        token_max_len=24,
        top_n=100,
        build_top_answers=False
    )
    test_data.top_answers = train_data.top_answers
    test_data.answer_to_idx = train_data.answer_to_idx
    test_data.idx_to_answer = train_data.idx_to_answer

    # Quick example to see how show() works
    # train_data.show(0, rich=True)

    # Create PyTorch Dataloaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate)
    val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False, collate_fn=custom_collate)
    test_loader  = DataLoader(test_data,  batch_size=16, shuffle=False, collate_fn=custom_collate)

    # Build model
    # We have top_n=100 => total classes = 101 (0..99 + 100 for <OTHER>)
    model = SimpleVQAModel(num_answers=101, freeze_bert=True, hidden_dim=256).to(device)

    # Losses
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 5
    for epoch in range(1, epochs+1):
        bin_loss, ans_loss = train_one_epoch(
            model, train_loader, optimizer, device, ce_loss, bce_loss
        )
        bin_acc, ans_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{epochs} - BinLoss: {bin_loss:.4f}, AnsLoss: {ans_loss:.4f}, "
              f"ValBinAcc: {bin_acc:.4f}, ValAnsAcc: {ans_acc:.4f}")

    # Evaluate with official VQA metric on val
    vqa_score = evaluate_vqa_official(model, val_loader, device, train_data.idx_to_answer)
    print(f"Validation Official VQA Score: {vqa_score:.4f}")

    # Generate submission on test set
    generate_submission(
        model,
        test_loader,
        device,
        idx_to_answer=train_data.idx_to_answer,
        submission_file="my_vizwiz_challenge2.json"
    )
    print("All done.")

if __name__ == "__main__":
    main()
