This repository contains the simplification code for the following paper:

Eliza Hobo, Charlotte Pouw, Lisa Beinborn (2023):<br>
"Geen makkie": Interpretable Classification and Simplification of Dutch Text Complexity <br>
to appear at the ACL-Workshop for [Innovative Use of NLP for Building Educational Applications](https://sig-edu.org/bea/2023)

The code for the readability experiments can be found [here](https://github.com/beinborn/InTeAM). 

We will also provide a link to the demo that is currently still undergoing security checks. 

### Simplification Experiments
The code used for the simplification experiments can be found in the **simplification folder**:
1) [`scripts`](simplification/scripts): Folder containing the used scripts
1) [`datasets`](simplification/datasets): Folder containing the used datasets
1) [`results`](simplification/results): Folder containing the produced results
1) [`models`](simplification/models): Folder for storing used embedding models

#### Deploying the Pipeline
Deployment of the simplification pipeline is done with the ```BERT_for_LS.py``` file and uses the following arguments:

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--model`| str| `the name of the model that is used for generating the predictions: a path to a folder or a huggingface directory.`|  -|
|`--eval_dir`| str| `path to the file with the to-be-simplified sentences.`| -|
|`--analysis`| Bool| `whether or not to output all the generated candidates and the reason for their removal `|False|
|`--ranking`| Bool| `whether or not to perform ranking of the generated candidates`|False|
|`--evaluation`| Bool| `whether or not to perform an evaluation of the generated candidates`|True|
|`--num_selections`| int| `the amount of candidates to generate`|10|
---

The pipeline can be run for English or Dutch:

**English**
1) Download a word embedding model from (fasttext) and store it in the models folder as __crawl-300d-2M-subword.vec__
1) Download the BenchLS, NNSeval and lex.mturk datasets from https://simpatico-project.com/?page_id=109 dataset and store them in the models folder

Then, the model can be run as follows:
```
python3 BERT_for_LS.py --model bert-large-uncased-whole-word-masking --eval_dir ../datasets/Dutch/dutch_data.txt 
```

**Dutch**

1) Download the word embedding model from https://dumps.wikimedia.org/nlwiki/20160501/ and store it in the models folder as __wikipedia-320.txt__

Then the model can be run as follows:
```
python3 BERT_for_LS.py --model GroNLP/bert-base-dutch-cased --eval_dir ../datasets/Dutch/dutch_data.tsv
```

#### Finetuning a Model
Two finetuning approaches:

1) Continual pre-training:
   ```
   python3 cpt_finetuning.py   --nr_sents 10000   
                               --epochs 2
                               --model_directory ../models/cpt_model
                               --seed 3
                               --language nl
   ```
   For English, the simple wikipedia corpus is used. 
   For Dutch, the Wablieft corpus is used.
   
1) Multi-task learning:
   ```
   python3 mtl-finetuning.py   --nr_sents 10000   
                               --epochs 2
                               --model_directory ../models/mtl_model
                               --seed 3
   ```
   For English, the simple-regular aligned wikipedia corpus is used.
