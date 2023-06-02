import numpy as np
import nltk
import torch
import spacy
from sklearn.metrics.pairwise import cosine_similarity as cosine
from scipy.special import softmax


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InputFeatures(object):
    """
    A set of features that together form the input to the model
    """

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def pipeline_evaluation(substitution_words, complex_words, annotated_subs):
    """
    :param substitution_words:
    :param complex_words:
    :param annotated_subs:
    :return:
    """

    # Todo : understand and comment this function

    instances = len(substitution_words)
    precision = 0
    accuracy = 0
    changed_proportion = 0

    for sub, source, gold in zip(substitution_words, complex_words, annotated_subs):
        if sub == source or (sub in gold):
            precision += 1
        if sub != source and (sub in gold):
            accuracy += 1
        if sub != source:
            changed_proportion += 1

    return precision / instances, accuracy / instances, changed_proportion / instances


def generation_evaluation(candidates_list, annotated_subs):
    """
    :param candidates_list:
    :param annotated_subs:
    :return:
    """
    assert len(candidates_list) == len(annotated_subs)

    potential = 0
    instances = len(candidates_list)
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0

    # Loop over the candidates
    for i in range(len(candidates_list)):

        # The words that are both in the candidates and the annotations:
        common = list(set(candidates_list[i]).intersection(annotated_subs[i]))

        # If there are some in common, potential is increased with 1
        if len(common) >= 1:
            potential += 1

        precision += len(common)
        recall += len(common)
        precision_all += len(candidates_list[i])
        recall_all += len(annotated_subs[i])

    # Potential is the percentage of time there was an overlap
    potential /= instances

    # Precision is the length of the overlap relative to the number of produced candidates
    precision /= precision_all

    # Recall is the length of the overlap relative to the number of annotated substitutions
    recall /= recall_all
    try:
        f_score = 2 * precision * recall / (precision + recall)
    except:
        f_score = 0

    return potential, precision, recall, f_score


def cross_entropy_word(prediction, position, input_id):
    """
    :param prediction: the prediction from BERT Todo: find out what is exactly happening
    :param position: the index of the masked word
    :param input_id: the input id of the masked word
    :return:
   """

    prediction = softmax(prediction, axis=1)
    loss = 0
    loss -= np.log10(prediction[position, input_id])
    return loss


def get_score(sentence, tokenizer, model):
    """
    :param sentence: the (part of the) sentence
    :param tokenizer: the BERT tokenizer
    :param model: the BERT model
    :return:
    """
    tokenize_input = tokenizer.tokenize(sentence)

    len_sen = len(tokenize_input)

    start_token = '[CLS]'
    separator_token = '[SEP]'

    # Input starts with CLS
    tokenize_input.insert(0, start_token)
    tokenize_input.append(separator_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokenize_input)

    sentence_loss = 0

    for i, word in enumerate(tokenize_input):

        if word == start_token or word == separator_token:
            continue

        original_word = tokenize_input[i]
        tokenize_input[i] = '[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        mask_input = mask_input.to(device)
        with torch.no_grad():
            output = model(mask_input)
            prediction_scores = output[0]

        word_loss = cross_entropy_word(prediction_scores[0].cpu().numpy(), i, input_ids[i])
        sentence_loss += word_loss
        tokenize_input[i] = original_word

    return np.exp(sentence_loss / len_sen)


def language_model_score(complex_word, complex_word_context, substitution_candidates, tokenizer, model):
    """
    :param complex_word: the complexed, masked, word
    :param complex_word_context: the context of the complex word
    :param substitution_candidates: the candidates for substitution
    :param tokenizer: the BERT tokenizer
    :param model: the BERT model
    :return:
    """

    sub_sentence = ''

    for context in complex_word_context:
        sub_sentence += context + " "

    sub_sentence = sub_sentence.strip()

    lm = []
    for candidate in substitution_candidates:
        candidate_sub_sentence = sub_sentence.replace(complex_word, candidate)

        score = get_score(candidate_sub_sentence, tokenizer, model)
        lm.append(score)

    return lm


def preprocess_subs_ranking(complex_word, generated_subs, embedding_dict, embedding_vector, word_count):
    """
    :param complex_word: the complex, masked word
    :param generated_subs: the generated simplifications
    :param embedding_dict: the words in the embedding model
    :param embedding_vector: the corresponding vectors in the embedding model
    :param word_count: the word frequencies
    :return:
    """

    selected_subs = []
    similarity_scores = []
    count_scores = []

    # Look up the embedding value of the complex word
    in_embedding = True
    if complex_word not in embedding_dict:
        in_embedding = False
    else:
        embedding_value = embedding_vector[embedding_dict.index(complex_word)].reshape(1, -1)

    # Iterating over the candidate substitutions and attribution values
    for sub in generated_subs:

        # If the substitution is not in the word_count dict, it is not taken into account
        if sub not in word_count:
            continue

        # Otherwise, the count feature is set to its count val
        else:
            sub_count = word_count[sub]

        # If there is an embedding value for the complex word and candidate word, the similarity is calculated
        if in_embedding:

            # If there is embedding value for complex word, but not for candidate word, candidate is discarded
            if sub not in embedding_dict:
                print("its not in the embedding_dict")
                continue

            token_embedding_index = embedding_dict.index(sub)
            similarity = cosine(embedding_value, embedding_vector[token_embedding_index].reshape(1, -1))
            similarity_scores.append(similarity)

        # If it has not been discarded, it is added to the selection
        selected_subs.append(sub)
        count_scores.append(sub_count)

    return selected_subs, similarity_scores, count_scores


def substitution_ranking(complex_word, complex_word_context, candidate_words, embedding_vocab, embedding_vectors,
                         word_count,
                         tokenizer, model, num_cands):
    """
    :param complex_word: the complex word that has been masked
    :param complex_word_context: the words in the context of the complex word
    :param candidate_words: the BERT-generated simplifications
    :param embedding_vocab: the words in the embedding model
    :param embedding_vectors: the corresponding vectors in the embedding model
    :param word_count: the frequency file
    :param tokenizer: the used BERT tokenizer
    :param model: the used BERT MLM
    :param num_cands: the number of predictions to write out
    :return: predicted_word:
    """

    # Do some preprocessing to select only good candidates
    global predicted_word
    substitution_candidates, similarity_scores, frequency_scores = preprocess_subs_ranking(complex_word,
                                                                                           candidate_words,
                                                                                           embedding_vocab,
                                                                                           embedding_vectors,
                                                                                           word_count)

    # If there are no candidates left, just return the complex word num_cands time
    if len(substitution_candidates) == 0:
        outputs = [complex_word]*num_cands

        return outputs

    # Cosine Similarity:
    if len(similarity_scores) > 0:
        seq = sorted(similarity_scores, reverse=True)
        similarity_rank = [seq.index(v) + 1 for v in
                           similarity_scores]  # This describes for each subs candidate the position in the ranking

    # Frequency Score
    sorted_count = sorted(frequency_scores, reverse=True)
    count_rank = [sorted_count.index(v) + 1 for v in
                  frequency_scores]  # This describes for each subs candidate the position in the ranking

    # Language Model Score
    lm_score = language_model_score(complex_word,
                                    complex_word_context,
                                    substitution_candidates,
                                    tokenizer,
                                    model)
    ranked_lm = sorted(lm_score)
    lm_rank = [ranked_lm.index(v) + 1 for v in lm_score]  # The position list of the lm scores

    # BERT Prediction Score
    bert_rank = []
    for i in range(len(substitution_candidates)):
        bert_rank.append(i + 1)

    # Total Rank:
    if len(similarity_scores) > 0:
        all_ranks = [bert + sis + count + LM for bert, sis, count, LM in zip(bert_rank,
                                                                             similarity_rank,
                                                                             count_rank,
                                                                             lm_rank)]
    else:
        all_ranks = [bert + count + LM for bert, count, LM in zip(bert_rank,
                                                                  count_rank,
                                                                  lm_rank)]

    # The final predictions are the ones with the lowest score:
    outputs = []
    for x in range(num_cands):
        if not all_ranks == []:
            lowest_score = min(all_ranks)
            predicted_index = all_ranks.index(lowest_score)
            all_ranks.remove(lowest_score)

            predicted_word = substitution_candidates[predicted_index]
            substitution_candidates.remove(predicted_word)

        outputs.append(predicted_word)

    return outputs


def substitution_selection(complex_word,
                           predicted_tokens,
                           ps,
                           analysis,
                           evaluation_file_name,
                           model_name,
                           selection_size=10):
    """
    :param complex_word: the complex, masked, target word
    :param predicted_tokens: 20 most likely tokens generated by BERT
    :param ps: the porter stemmer
    :param evaluation_file_name: the directory to the data file (used to name the output file)
    :param model_name: name of the used model (also used to name the output file)
    :param analysis: whether or not to output stats on the filtering
    :param selection_size: the number of likely substitutions to return
    :return:
    """
    selected_tokens = []
    filtered_out_tokens = []

    subwords = 0
    actual_words = 0
    stem = 0
    too_close = 0

    # Find the stem of the complex word
    complex_word_stem = ps.stem(complex_word)

    assert selection_size <= len(predicted_tokens)

    # Loop over all predicted tokens
    for i in range(len(predicted_tokens)):
        predicted_token = predicted_tokens[i]

        # If BERT predicts a subword, it is not taken into account
        if predicted_token[0:2] == "##":
            subwords += 1
            filtered_out_tokens.append(predicted_token)
            continue

        # If BERT predicts the actual word, it is not taken into account
        if predicted_token == complex_word:
            filtered_out_tokens.append(predicted_token)
            actual_words += 1
            continue

        # If the stem of the predicted word is the same as that of the actual, it is not taken in to account
        predicted_token_stem = ps.stem(predicted_token)
        if predicted_token_stem == complex_word_stem:
            filtered_out_tokens.append(predicted_token)
            stem += 1
            continue

        # If the predicted token is very similar to the actual, it is not taken into account
        if (len(predicted_token_stem) >= 3) and (predicted_token_stem[:3] == complex_word_stem[:3]):
            filtered_out_tokens.append(predicted_token)
            too_close += 1
            continue

        # If the predicted token is not a subword and is different enough, it is added to the actual
        selected_tokens.append(predicted_token)

        # If enough tokens have been deleted, it's enough
        if len(selected_tokens) == selection_size:
            break

    # If none are good enough for the criteria, the first ones until the selection size are chosen
    if len(selected_tokens) == 0:
        selected_tokens = predicted_tokens[0:selection_size + 1]

    if analysis:
        with open(f"../{evaluation_file_name}_{model_name}_selections.txt", "a") as analysis_file:
            analysis_file.write(str(subwords) +
                                "\t" +
                                str(actual_words) +
                                "\t" +
                                str(stem) +
                                "\t" +
                                str(too_close) +
                                "\t")

            for filtered_out_token in filtered_out_tokens:
                analysis_file.write(filtered_out_token + "\t")
            analysis_file.write("\n")

    assert len(selected_tokens) > 0

    return selected_tokens


def substitution_selection_lemmatized(complex_word,
                                      predicted_tokens,
                                      ps,
                                      analysis,
                                      evaluation_file_name,
                                      model_name,
                                      nlp,
                                      selection_size=10):
    """
        :param complex_word: the complex, masked, target word
        :param predicted_tokens: 20 most likely tokens generated by BERT
        :param ps: the porter stemmer
        :param evaluation_file_name: the directory to the data file (used to name the output file)
        :param model_name: name of the used model (also used to name the output file)
        :param analysis: whether or not to output stats on the filtering
        :param nlp: the spacy model used for lemmatizing
        :param selection_size: the number of likely substitutions to return
        :return:
        """

    selected_tokens = []
    filtered_out_tokens = []

    subwords = 0
    actual_words = 0
    stem = 0
    too_close = 0

    # Find the stem of the complex word
    complex_word_stem = ps.stem(complex_word)
    lemmatized = False

    doc = nlp(complex_word)

    for token in doc:
        if token.text == complex_word:
            complex_word_stem = token.lemma_
            lemmatized = True

    assert selection_size <= len(predicted_tokens)

    # Loop over all predicted tokens
    for i in range(len(predicted_tokens)):
        predicted_token = predicted_tokens[i]

        # If BERT predicts a subword, it is not taken into account
        if predicted_token[0:2] == "##":
            subwords += 1
            filtered_out_tokens.append(predicted_token)
            continue

        # If BERT predicts the actual word, it is not taken into account
        if predicted_token == complex_word:
            filtered_out_tokens.append(predicted_token)
            actual_words += 1
            continue

        # If the stem of the predicted word is the same as that of the actual, it is not taken in to account
        predicted_token_stem = ps.stem(predicted_token)

        if lemmatized:
            doc = nlp(predicted_token)
            for token in doc:
                if token.text == predicted_token:
                    predicted_token_stem = token.lemma_

        if predicted_token_stem == complex_word_stem:
            filtered_out_tokens.append(predicted_token)
            stem += 1
            continue

        # If the predicted token is very similar to the actual, it is not taken into account
        if (len(predicted_token_stem) >= 3) and (predicted_token_stem[:3] == complex_word_stem[:3]):
            filtered_out_tokens.append(predicted_token)
            too_close += 1
            continue

        # If the predicted token is not a subword and is different enough, it is added to the actual
        selected_tokens.append(predicted_token)

        # If enough tokens have been deleted, it's enough
        if len(selected_tokens) == selection_size:
            break

    # If none are good enough for the criteria, the first ones until the selection size are chosen
    if len(selected_tokens) == 0:
        selected_tokens = predicted_tokens[0:selection_size + 1]

    if analysis:
        with open(f"../{evaluation_file_name}_{model_name}_selections.txt", "a") as analysis_file:
            analysis_file.write(str(subwords) + "\t" +
                                str(actual_words) + "\t" +
                                str(stem) + "\t" +
                                str(too_close)+"\t")
            for filtered_out_token in filtered_out_tokens:
                analysis_file.write(filtered_out_token + "\t")
            analysis_file.write("\n")

    assert len(selected_tokens) > 0

    return selected_tokens


def convert_token_to_feature(final_tokens, mask_position, seq_length, tokenizer):
    """
   If a single nltk token is tokenized into a single token by BERT,
   this function transforms a data file into a list of `InputFeature`s.
   :param final_tokens: [CLS] sentence [SEP] masked sentence [CLS]
   :param mask_position: index of the complex word/ mask
   :type: mask_position: list
   :param seq_length: maximum length of BERT sequence
   :param tokenizer: used BERT tokenizer
   :return:
   """

    tokens = ["[CLS]"]      # This will be filled with all tokens
    input_type_ids = [0]

    for token in final_tokens:
        tokens.append(token)    # Fill in with the original sentence
        input_type_ids.append(0)    # And a corresponding list for the identification of the separate sents

    tokens.append("[SEP]")      # The sentence ends with [SEP]
    input_type_ids.append(0)    # And the identification of the separate sents

    for token in final_tokens:
        tokens.append(token)    # Add them a second time
        input_type_ids.append(1)    # But then with 1s

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    if isinstance(mask_position, list):
        for pos in mask_position:
            true_word = true_word + tokens[pos]
            tokens[pos] = '[MASK]'
    else:
        tokens[mask_position] = '[MASK]'

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    # Check that they are all the same length
    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def convert_whole_word_to_feature(bert_sent, mask_position, seq_length, tokenizer):
    """
    If a single nltk token is tokenized into multiple subwords for BERT,
    this function transforms a data file into a list of `InputFeature`s.
    :param bert_sent: [CLS] sentence [SEP] masked sentence [CLS]
    :param mask_position: index of the complex word/ mask
    :type: mask_position: list
    :param seq_length: maximum length of BERT sequence
    :param tokenizer: used BERT tokenizer
    :return:
    """

    final_tokens = ["[CLS]"]    # This will be filled with all tokens
    input_type_ids = [0]
    for token in bert_sent:     # Fill in with the original sentence
        final_tokens.append(token)
        input_type_ids.append(0)    # And a corresponding list for the identification of the separate sents

    final_tokens.append("[SEP]")    # The sentence ends with [SEP]
    input_type_ids.append(0)        # And the identification of the separate sents

    for token in bert_sent:
        final_tokens.append(token)  # Add them a second time
        input_type_ids.append(1)    # But then with 1s

    final_tokens.append("[SEP]")
    input_type_ids.append(1)

    # The number of subwords that make up the token:
    len_masked_subwords = len(mask_position)
    count = 0

    # Replacing all subwords with one [MASK] token
    while count in range(len_masked_subwords):
        index = len_masked_subwords - 1 - count

        pos = mask_position[index]

        if index == 0:
            final_tokens[pos] = '[MASK]'

        else:
            del final_tokens[pos]
            del input_type_ids[pos]

        count += 1

    # Convert the final tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    # Check that they are all the same length
    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=final_tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def extract_context(words, mask_index, window):
    """
    This function extracts the context of the complex word in a sentence
    :param words: nltk tokenized sentence
    :type words: list
    :param mask_index: index of the complex word
    :param window: size of context window
    :type window: int
    :return: context: the words surrounding the complex words
    """

    sent_length = len(words)

    half_window = int(window / 2)

    # Check that the complex word is located inside the sentence
    assert mask_index >= 0 and mask_index < sent_length

    context = ""

    # If the sentence is shorter than the window, the whole sentence is returned
    if sent_length <= window:
        context = words

    # if the mask is in the first half and the
    elif mask_index < sent_length - half_window and mask_index >= half_window:
        context = words[mask_index - half_window: mask_index + half_window + 1]

    elif mask_index < half_window:
        context = words[0:window]

    elif mask_index >= sent_length - half_window:
        context = words[sent_length - window:sent_length]

    else:
        print("Wrong!")

    return context


def convert_sentence_to_token(sentence, seq_length, tokenizer, lower_case):
    # I think the general purpose of this function is to align the raw texts with the BERT tokenizer?
    if lower_case:
        sentence = sentence.lower()

    # Use BERT tokenizer to tokenize text
    tokenized_text = tokenizer.tokenize(sentence)

    # The tokenized text must be smaller than the maximal length-2 (with the sep/CLS token?)
    assert len(tokenized_text) < seq_length - 2

    # Then tokenize the sentence with nltk
    nltk_sent = nltk.word_tokenize(sentence)

    position2 = []

    token_index = 0

    # Why would you start
    start_pos = len(tokenized_text) + 2

    pre_word = ""

    # Loop over the nltk tokenized words, make some corrections
    for i, word in enumerate(nltk_sent):

        if word == "n't" and pre_word[-1] == "n":
            word = "'t"

        if tokenized_text[token_index] == "\"":
            len_token = 2
        else:
            len_token = len(tokenized_text[token_index])

        if tokenized_text[token_index] == word or len_token >= len(word):
            position2.append(start_pos + token_index)
            pre_word = tokenized_text[token_index]

            token_index += 1
        else:
            new_pos = []
            new_pos.append(start_pos + token_index)

            new_word = tokenized_text[token_index]

            while new_word != word:

                token_index += 1

                new_word += tokenized_text[token_index].replace('##', '')

                new_pos.append(start_pos + token_index)

                if len(new_word) == len(word):
                    break
            token_index += 1
            pre_word = new_word

            position2.append(new_pos)

    return tokenized_text, nltk_sent, position2


def read_eval_index_dataset(data_path):
    """
    Function to read in the BenchLS, NNSeval and Dutch data sets.
    :param data_path: location of the  file
    :return: sentences: list of sentences
    :return: complex_words: list of the complex words
    :return: substitutions: list of lists of the annotated simplifications
    """

    sentences = []
    complex_words = []
    all_annotations = []

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()

            if not line:
                break

            # Collect the sentences and words
            sentence, words = line.strip().split('\t', 1)

            # Split the words into the complex word and possible simplifications
            complex_word, annotations = words.strip().split('\t', 1)
            annotation_list = annotations.split('\t')

            # Remove the numbers from the annotations

            annotations_without_numbs = []
            for anno in annotation_list[1:]:
                anno_id, anno_word = anno.split(':')
                annotations_without_numbs.append(anno_word)

            annotations = list(set(annotations_without_numbs))
            all_annotations.append(annotations)

            sentences.append(sentence)
            complex_words.append(complex_word)

    return sentences, complex_words, all_annotations


def read_eval_dataset_lexmturk(data_path):
    """
    Function to read in the "lex.mturk.txt" data set.
    :param: data_path: location of the lex.mturk.txt file
    :return: sentences: list of sentences
    :return: complex_words: list of the complex words
    :return: substitutions: list of lists of the annotated simplifications
    """
    sentences = []
    complex_words = []
    all_annotations = []
    line_nr = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            line_nr += 1

            # Skip the header
            if line_nr == 1:
                continue

            if not line:
                break

            # Retrieve the sentence, complex words and the annotations
            sentence, words = line.strip().split('\t', 1)
            complex_word, annotations = words.strip().split('\t', 1)
            annotation_list = annotations.split('\t')

            # Removing duplicates from the annotations
            annotations = list(set(annotation_list))

            sentences.append(sentence)
            complex_words.append(complex_word)
            all_annotations.append(annotations)

    return sentences, complex_words, all_annotations


def read_eval_dataset_dutch(data_path):
    """
    Function to read in the "lex.mturk.txt" data set.
    :param: data_path: location of the lex.mturk.txt file
    :return: sentences: list of sentences
    :return: complex_words: list of the complex words
    :return: substitutions: list of lists of the annotated simplifications
    """
    sentences = []
    complex_words = []
    all_annotations = []
    line_nr = 0

    with open(data_path, "r", encoding='UTF-8') as reader:
        while True:
            line = reader.readline()
            line_nr += 1

            if not line:
                break

            # Retrieve the sentence, complex words and the annotations
            sentence, words = line.strip().split('\t', 1)
            # print(sentence)
            # print(words)
            complex_word, annotations = words.strip().split('\t', 1)
            annotation_list = annotations.split('\t')

            # Removing duplicates from the annotations
            annotations = list(set(annotation_list))

            sentences.append(sentence)
            complex_words.append(complex_word)
            all_annotations.append(annotations)

    return sentences, complex_words, all_annotations


def get_words_and_vectors(embedding_path):
    """
    This function creates a list of all words, and a list of all embedding vectors
    :param embedding_path: location of word embedding model
    :returns words: list of all words in word embedding model
    :returns vectors: list of corresponding word vectors
    """
    words = []
    vectors = []
    f = open(embedding_path, 'r', encoding="utf-8")
    lines = f.readlines()

    for (n, line) in enumerate(lines):

        if n == 0:
            # The first line describes the embedding dimension
            print("Word embedding of size: ", line)
            continue

        # The rest of the lines contain the word -space- the vector
        word, vector = line.rstrip().split(' ', 1)
        vector = np.fromstring(vector, sep=' ')
        vectors.append(vector)
        words.append(word)

    f.close()
    return words, vectors


def get_word_count(word_count_path):
    """
    :param word_count_path: location of the frequency file
    :return word2count: dictionary of words and their frequency
    :rtype: dict
    """
    # Makes a dictionary word : freq
    word2count = {}
    with open(word_count_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            i = i.strip()
            if len(i) > 0:
                i = i.split()
                if len(i) == 2:
                    word2count[i[0]] = float(i[1])
                else:
                    print(i)
    return word2count
