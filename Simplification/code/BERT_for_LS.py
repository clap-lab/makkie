import logging
import argparse
import spacy

from nltk import PorterStemmer
from nltk.stem import SnowballStemmer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils2 import *
from transformers import BertForPreTraining, BertTokenizer

# Run this line the first time:
# python -m spacy download nl_core_news_sm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(my_args=None):
    """
    Parsing the input and running the functions
    """

    parser = argparse.ArgumentParser()
    global complex_words

    if my_args is None:

        # Model to be run:
        parser.add_argument("--model",
                            default=None,
                            required=True,
                            help="The language of the model")

        # Directory of evaluation data (BenchLS/ Lexmturk/ NNSeval/ Dutch)
        parser.add_argument("--eval_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The evaluation data directory.")

        # The maximum total input sequence length after WordPiece tokenization
        parser.add_argument("--max_seq_length",
                            default=250,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")

        # Number of generated simplifications
        parser.add_argument("--num_selections",
                            default=10,
                            type=int,
                            help="Total number of generated simplifications.")

        parser.add_argument("--analysis",
                            default=False,
                            type=bool,
                            required=False,
                            help="Whether to output analysis information.")

        parser.add_argument("--ranking",
                            default=False,
                            type=bool,
                            help="Whether to perform the ranking procedure as well")

        parser.add_argument("--evaluation",
                            default=True,
                            type=bool,
                            help="Whether to calculate performance scores")

        # Parsing Command line input
        args = parser.parse_args()

        used_model = args.model
        eval_dir = args.eval_dir
        evaluation_file_name = args.eval_dir.split('/')[-1][:-4]  # Evaluation file:
        max_seq_length = args.max_seq_length
        num_selections = args.num_selections     # Number of candidates for substitution
        # results_file = open(args.results_file, "a+")
        # out_file = args.out_file
        analysis = args.analysis
        ranking = args.ranking
        evaluation = args.evaluation

    else:
        # Parsing internal input
        used_model = my_args[0]
        eval_dir = my_args[1]
        evaluation_file_name = eval_dir.split('/')[-1][:-4]  # Evaluation file:
        max_seq_length = int(my_args[2])
        num_selections = int(my_args[3])    # Number of candidates for substitution
        # results_file = open(my_args[4], "a+")
        # out_file = my_args[5]
        analysis = bool(my_args[4])
        ranking = bool(my_args[5])
        evaluation = bool(my_args[6])

    model_name = used_model.replace("../models/", "").replace("/", "")

    results_file = open(f"../results/{evaluation_file_name}_{model_name}_results.txt", "a")
    out_file = f"../results/{evaluation_file_name}_{model_name}_outputs.txt"

    # Loading in the Model & Tokenizer
    logger.info("Loading the model and tokenizer")

    if used_model.startswith("bert"):  # If it is from the huggingface library
        logger.info("you are using a model from the huggingface library")
        tokenizer = AutoTokenizer.from_pretrained(used_model)
        model = AutoModelForMaskedLM.from_pretrained(used_model)

    else:  # If it was finetuned
        logger.info("you are using a custom model")
        model = BertForPreTraining.from_pretrained(used_model)
        tokenizer = BertTokenizer.from_pretrained(used_model)

    if "dutch" in used_model:
        logger.info("you are using a dutch model")
        lower_case = True
        stemmer = SnowballStemmer("dutch")
        lemmatizer = True
        nlp = spacy.load("nl_core_news_sm")
    else:
        lower_case = True
        stemmer = PorterStemmer()  # Loading the stemmer
        lemmatizer = False
        # nlp = spacy.load("nl_core_news_sm")

    if ranking:
        if "dutch" in used_model:  # If it's a dutch model, it needs the dutch files
            logger.info("you are using a Dutch model")
            embedding_path = "../models/wikipedia-320.txt"
            word_count_path = "../datasets/dutch_frequencies.txt"

        else:  # And the English for the English
            embedding_path = "../models/crawl-300d-2M-subword.vec"
            word_count_path = "../datasets/frequency_merge_wiki_child.txt"

        # Loading in Embeddings
        logger.info("Loading embeddings ...")

        word_count = get_word_count(word_count_path)  # This is a dictionary with the shape word: frequency

        # vocabulary of embedding model in a list, and corresponding emb values in another
        embedding_vocab, embedding_vectors = get_words_and_vectors(embedding_path)
        logger.info("Done loading embeddings!")
    # Toward Generating the Substitutions:
    candidates_list = []
    final_predictions = []
    window_context = 11

    # Retrieve the sentences, complex words and annotated labels
    if evaluation_file_name == 'lex.mturk':  # Specifically for these files (with header etc)
        eval_sents, complex_words, annotated_subs = read_eval_dataset_lexmturk(eval_dir)
    elif "dutch" in evaluation_file_name:
        eval_sents, complex_words, annotated_subs = read_eval_dataset_dutch(eval_dir)
    else:
        eval_sents, complex_words, annotated_subs = read_eval_index_dataset(eval_dir)

    # Starting the evaluation
    logger.info("***** Running evaluation *****")

    # Pytorch model in evaluation mode:
    model.eval()

    eval_size = len(eval_sents)

    # Loop over the evaluation sentences:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # Location of execution:
    model.to(device)

    with open(out_file, "w", encoding="UTF-8") as outfile:

        for i in range(eval_size):
            logger.info(f"***** Sentence {i} of {eval_size} *****")
            sentence = eval_sents[i]

            logger.info(f"sentence: {sentence}")

            if lower_case:
                sentence = sentence.lower()

            outfile.write(str(sentence)+"\t")

            # Making a mapping between BERT's subword tokenized sent and nltk tokenized sent
            bert_sent, nltk_sent, bert_token_positions = convert_sentence_to_token(sentence, max_seq_length,
                                                                                   tokenizer, lower_case)
            logger.info(f"bert sent:{bert_sent}")
            logger.info(f"nltk sent:{nltk_sent}")
            # Check alignment
            assert len(nltk_sent) == len(bert_token_positions)

            if lower_case:
                complex_word = complex_words[i].lower()
            else:
                complex_word = complex_words[i]

            logger.info(f"complex word: {complex_word}")

            mask_index = nltk_sent.index(complex_word)  # the location of the complex word:
            outfile.write(nltk_sent[mask_index])

            mask_context = extract_context(nltk_sent, mask_index, window_context)  # the words surrounding it

            bert_mask_position = bert_token_positions[mask_index]  # The BERT index of [MASK]

            if isinstance(bert_mask_position, list):  # If the mask is at a sub-word-tokenized token
                # This is an instance of the feature class
                feature = convert_whole_word_to_feature(bert_sent, bert_mask_position, max_seq_length, tokenizer)
            else:
                feature = convert_token_to_feature(bert_sent, bert_mask_position, max_seq_length, tokenizer)

            # Turn it into tensors
            tokens_tensor = torch.tensor([feature.input_ids])
            token_type_ids = torch.tensor([feature.input_type_ids])
            attention_mask = torch.tensor([feature.input_mask])

            # And on their way to the CUDA/ CPU
            tokens_tensor = tokens_tensor.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Make predictions
            with torch.no_grad():
                output = model(tokens_tensor, attention_mask=attention_mask, token_type_ids=token_type_ids)
                prediction_scores = output[0]

            if isinstance(bert_mask_position, list):
                predicted_top = prediction_scores[0, bert_mask_position[0]].topk(num_selections * 2)
            else:
                predicted_top = prediction_scores[0, bert_mask_position].topk(num_selections * 2)

            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1])

            # A hard cut on the selsection, leaving maximum num_selection candidates

            if not lemmatizer:

                candidate_words = substitution_selection(complex_word,
                                                         predicted_tokens,
                                                         stemmer,
                                                         analysis,
                                                         evaluation_file_name,
                                                         model_name,
                                                         num_selections
                                                         )
            else:
                candidate_words = substitution_selection_lemmatized(complex_word,
                                                                    predicted_tokens,
                                                                    stemmer,
                                                                    analysis,
                                                                    evaluation_file_name,
                                                                    model_name,
                                                                    nlp,
                                                                    num_selections
                                                                    )
            candidates_list.append(candidate_words)

            if ranking:

                highest_predictions = substitution_ranking(complex_word,
                                                           mask_context,
                                                           candidate_words,
                                                           embedding_vocab,
                                                           embedding_vectors,
                                                           word_count,
                                                           tokenizer,
                                                           model,
                                                           num_selections)
            else:
                highest_predictions = candidate_words

            for word in highest_predictions:
                outfile.write("\t" + word)
            outfile.write("\n")

            predicted_word = highest_predictions[0]
            final_predictions.append(predicted_word)

    if evaluation:
        potential, precision, recall, f_score = generation_evaluation(candidates_list, annotated_subs)
        print("The score of evaluation for substitution selection")
        results_file.write(str(used_model))
        results_file.write(str(num_selections))
        results_file.write('\t')
        results_file.write(str(potential))
        results_file.write('\t')
        results_file.write(str(precision))
        results_file.write('\t')
        results_file.write(str(recall))
        results_file.write('\t')
        results_file.write(str(f_score))
        results_file.write('\t')
        print(potential, precision, recall, f_score)

        if ranking:
            precision, accuracy, changed_proportion = pipeline_evaluation(final_predictions,
                                                                      complex_words,
                                                                      annotated_subs)
            print("The score of evaluation for full LS pipeline")
            print(precision, accuracy, changed_proportion)
            results_file.write(str(precision))
            results_file.write('\t')
            results_file.write(str(accuracy))
            results_file.write('\t')
            results_file.write(str(changed_proportion))
            results_file.write('\n')

        results_file.close()


if __name__ == "__main__":
    main()
