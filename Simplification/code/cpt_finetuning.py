import os
import random
import torch
import argparse
import pandas as pd
import logging

from transformers import BertTokenizerFast
from transformers import BertForPreTraining, BertTokenizer, BertForMaskedLM
from transformers import AdamW
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def create_inputs(tokenizer, sentence_a, sentence_b, label):

    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                       max_length=250, truncation=True, padding='max_length')

    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    inputs['labels'] = inputs.input_ids.detach().clone()

    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    return inputs


def get_classification_instances(pages, num_sents):
    sentence_a = []
    sentence_b = []
    label = []

    text = ""
    for page in pages.values():
        text += page

    bag = [item for sentence in text for item in sentence.split('.') if item != '']
    bag_size = len(bag)
    i=0
    for title, text in pages.items():
        if i>num_sents:
            break
        i+=1
        sentences = [sentence for sentence in text.split('.') if sentence != '']
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences - 2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start + 1])
                label.append(0)
            else:
                index = random.randint(0, bag_size - 1)
                # this is NotNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(bag[index])
                label.append(1)

    return sentence_a, sentence_b, label

def get_Dutch_data(num_sents, level):

    path = "../datasets/wablief_sents"

    pages = dict()

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            pages[filename] = ""
            with open (f,"r") as infile:
                data = infile.readlines()
            for row in data:
                row = row.strip()
                pages[filename] += row
    print(pages[filename])
    return pages

def get_data(num_sents):
    with open("../datasets/wikisimple.txt", "r") as infile:
        data = infile.readlines()

    pages = dict()
    new_page = True

    for row in data:
        row = row.strip()

        if new_page:
            subject = row
            pages[subject] = ""
            new_page = False

        elif row == "":
            new_page = True

        else:
            pages[subject] = pages[subject] + row

    return pages


def main(my_args=None):

    if my_args is None:
        parser = argparse.ArgumentParser()

        parser.add_argument("--nr_sents",
                            default=10000,
                            type=int,
                            required=False,
                            help="Number of sentences")

        parser.add_argument("--lr",
                            default=5e-6,
                            type=float,
                            required=False,
                            help="The learning rate")

        parser.add_argument("--epochs",
                            default=2,
                            type=float,
                            required=False,
                            help="The number of epochs")

        parser.add_argument("--model_directory",
                            type=str,
                            required=True,
                            help="Directory to store model")

        parser.add_argument("--random_seed",
                            type=int,
                            default=1,
                            required=False,
                            help="Directory to store model")

        parser.add_argument("--language",
                           default="English",
                           type=str,
                           required=True,
                           help="To finetune the English or the Dutch model")

        parser.add_argument("--level",
                           default="Accepted",
                           type=str,
                           required=False,
                           help="The level of the dutch texts")
        args = parser.parse_args()

        num_sents = args.nr_sents
        nr_epochs = args.epochs
        learning_rate = args.lr
        model_dir = args.model_directory
        seed = args.random_seed
        language = args.language
        level = args.level

    else:
        num_sents = my_args[0]
        nr_epochs = my_args[1]
        learning_rate = my_args[2]
        model_dir = my_args[3]
        seed = my_args[4]
        language = my_args[5]
        level = my_args[6]

    if language == "English":
        logger.info("you are training an English model")
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        model = BertForPreTraining.from_pretrained('bert-large-uncased-whole-word-masking')

    if language == "Dutch":
        logger.info("you are training a Dutch model")
        tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
        model = BertForPreTraining.from_pretrained('GroNLP/bert-base-dutch-cased')

    random.seed(seed)
    torch.manual_seed(seed)

    if language == "English":
        text = get_data(num_sents)

    if language == "Dutch":
        text = get_Dutch_data(num_sents, level)

    # pages = get_data(num_sents)

    sentence_a, sentence_b, label = get_classification_instances(text, num_sents)
    print("length of input:", len(sentence_a))

    inputs = create_inputs(tokenizer, sentence_a, sentence_b, label)
    dataset = OurDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(nr_epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=False)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            next_sentence_label = batch['next_sentence_label'].to(device)

            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    # Create output directory if needed
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    main()