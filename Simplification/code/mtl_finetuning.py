import os
import random
import torch
import argparse
import pandas as pd

from transformers import BertTokenizerFast
from transformers import BertForPreTraining
from transformers import AdamW
from tqdm import tqdm


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


def get_classification_instances(normal_sents, simple_sents):
    sentence_a = []
    sentence_b = []
    label = []

    for normal_sent, simple_sent in zip(normal_sents, simple_sents):

        random_number = random.random()

        if random_number > 0.5:
            sentence_a.append(normal_sent)
            sentence_b.append(simple_sent)
            label.append(1)
        else:
            sentence_a.append(simple_sent)
            sentence_b.append(normal_sent)
            label.append(0)

    return sentence_a, sentence_b, label


def get_data(num_sents):
    normal = pd.read_csv("../datasets/Wikipedia simple/normal.aligned", sep="\t",
                         names=["subject", "nr", "sentence"])
    simple = pd.read_csv("../datasets/Wikipedia simple/simple.aligned", sep="\t",
                         names=["subject_simple", "nr_simple", "sentence_simple"])
    combination = pd.concat([simple, normal], axis=1, join="inner")
    combination = combination.sample(frac=1, random_state=1)
    combination.drop(['nr', 'subject', 'nr_simple', 'subject_simple'], axis=1)
    selection = combination[:num_sents]
    normal_sents = selection['sentence'].tolist()
    simple_sents = selection['sentence_simple'].tolist()

    return normal_sents, simple_sents


def main(my_args=None):

    if my_args is None:
        parser = argparse.ArgumentParser()

        parser.add_argument("--nr_sents",
                            default=10000,
                            type=int,
                            required=True,
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

        args = parser.parse_args()

        num_sents = args.nr_sents
        nr_epochs = args.epochs
        learning_rate = args.lr
        model_dir = args.model_directory
        seed = args.random_seed

    else:
        num_sents = my_args[0]
        nr_epochs = my_args[1]
        learning_rate = my_args[2]
        model_dir = my_args[3]
        seed = my_args[4]

    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking')
    normal_sents, simple_sents = get_data(num_sents)

    random.seed(seed)
    torch.manual_seed(seed)

    sentence_a, sentence_b, label = get_classification_instances(normal_sents, simple_sents)
    inputs = create_inputs(tokenizer, sentence_a, sentence_b, label)
    dataset = OurDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    model = BertForPreTraining.from_pretrained('bert-large-uncased-whole-word-masking')
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