"""
Based on this article: https://arxiv.org/pdf/1708.04439.pdf
"""
import collections
import logging
import math
import os
import re
import sys

import numpy as np
from sklearn.neural_network import BernoulliRBM
import xml.etree.ElementTree as ET

import czech_stemmer
import rbm
from RDRPOSTagger_python_3.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from RDRPOSTagger_python_3.Utility.Utils import readDictionary
os.chdir('../..')  # because above modules do chdir ... :/
from rouge_2_0.rouge_20 import print_rouge_scores
import separator
import textrank

logger = logging.getLogger('summarizer')
logging.basicConfig(level=logging.DEBUG)

STOPWORDS = set()
with open('stopwords.txt', 'r') as f:
    for w in f:
        STOPWORDS.add(w.strip())


def pos_tag(sentences):
    r = RDRPOSTagger()
    # Load the POS tagging model
    r.constructSCRDRtreeFromRDRfile('./RDRPOSTagger_python_3/Models/UniPOS/UD_Czech-CAC/train.UniPOS.RDR')
    # Load the lexicon
    rdr_pos_dict = readDictionary('./RDRPOSTagger_python_3/Models/UniPOS/UD_Czech-CAC/train.UniPOS.DICT')
    tagged_sentences = []
    for sentence in sentences:
        tagged_sentence_orig = r.tagRawSentence(rdr_pos_dict, sentence)
        tagged_words = tagged_sentence_orig.split()
        tagged_sentence = []
        for t_w in tagged_words:
            word, tag = t_w.split('/')
            tagged_sentence.append((word, tag))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences


def remove_stop_words(sentences, keep_case=False, is_tokenized=True, return_tokenized=True):
    if is_tokenized:
        tokenized_sentences = sentences
    else:
        tokenized_sentences = tokenize(sentences)
    sentences_without_stopwords = []
    for sentence_orig in tokenized_sentences:
        sentence_without_stopwords = []
        for word in sentence_orig:
            if word.lower() not in STOPWORDS:
                sentence_without_stopwords.append(word if keep_case else word.lower())
        sentences_without_stopwords.append(
            sentence_without_stopwords if return_tokenized else ' '.join(sentence_without_stopwords)
        )
    return sentences_without_stopwords


def tokenize(sentences):
    tokenized = []
    for s in sentences:
        tokenized.append([w.strip(' ,.!?"():;-') for w in s.split()])
    return tokenized


def thematicity_feature(tokenized_sentences, most_common_cutoff=10):
    words = [word for sentence in tokenized_sentences for word in sentence]
    counts = collections.Counter(words)
    most_common = counts.most_common(most_common_cutoff)
    thematic_words = []
    for word, _ in most_common:
        thematic_words.append(word)
    logger.debug(f'Thematic words: {thematic_words}')
    thematicity_scores = []
    for sentence in tokenized_sentences:
        count_of_thematic_words = 0
        for word in sentence:
            if word in thematic_words:
                count_of_thematic_words += 1
        thematicity = count_of_thematic_words / (len(sentence) + 0.000001)
        thematicity_scores.append(thematicity)
    max_score = max(thematicity_scores)
    thematicity_scores = [score / max_score for score in thematicity_scores]
    return thematicity_scores


def upper_case_feature(tokenized_sentences):
    tokenized_sentences_wo_sw = remove_stop_words(tokenized_sentences, keep_case=True)
    scores = []
    for sentence in tokenized_sentences_wo_sw:
        count_of_uppercase_starting_words = 0
        for word in sentence:
            if word[0].isupper():
                count_of_uppercase_starting_words += 1
        scores.append(count_of_uppercase_starting_words / (len(sentence) + 0.000001))
    return scores


def tf_isf_orig_feature(tokenized_sentences):
    scores = []
    words = [word for sentence in tokenized_sentences for word in sentence]
    counts_total = collections.Counter(words)
    for sentence in tokenized_sentences:
        counts = collections.Counter(sentence)
        score = 0
        for word in counts.keys():
            score += math.log(counts[word] * counts_total[word])
        scores.append(score / (len(sentence) + 0.000001))
    return scores


# def tf_isf_feature(tokenized_sentences):
#     scores = []
#     for sentence in tokenized_sentences:
#         counts = collections.Counter(sentence)
#         tf_isf = 0
#         for word in counts.keys():
#             sentences_with_word_count = 0
#             for sentence_2 in tokenized_sentences:
#                 if word in sentence_2:
#                     sentences_with_word_count += 1
#             tf_isf += counts[word] * math.log(len(tokenized_sentences) / sentences_with_word_count)
#         scores.append(tf_isf / (len(sentence) + 0.000001))
#     return scores


def proper_noun_feature(tagged):
    scores = []
    for sentence in tagged:
        score = 0
        for word, tag in sentence:
            if tag == 'PROPN':
                score += 1
        scores.append(score / (len(sentence) + 0.000001))
    return scores


def text_to_word_counter(text):
    word_re = re.compile(r'\w+')
    words = word_re.findall(text)
    return collections.Counter(words)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0
    else:
        return numerator / denominator


def centroid_similarity_feature(sentences, tf_isf_scores):
    scores = []
    centroid_index = tf_isf_scores.index(max(tf_isf_scores))
    vector_1 = text_to_word_counter(sentences[centroid_index])
    for sentence in sentences:
        vector_2 = text_to_word_counter(sentence)
        score = get_cosine(vector_1, vector_2)
        scores.append(score)
    return scores


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def numerals_feature(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences:
        score = 0
        for word in sentence:
            if is_number(word):
                score +=1
        scores.append(score / (len(sentence) + 0.000001))
    return scores


# as originally defined, but doesn't work very well
# def sentence_position_feature(num_sentences):
#     threshold = 0.2 * num_sentences
#     min_v = threshold * num_sentences
#     max_v = threshold * 2 * num_sentences
#     pos = []
#     for sentence_pos in range(num_sentences):
#         if sentence_pos in (0, num_sentences - 1):
#             pos.append(1)
#         else:
#             t = math.cos((sentence_pos - min_v) * ((1 / max_v) - min_v))
#             pos.append(t)
#     return pos


def sentence_position_feature(num_sentences):
    pos = []
    for sentence_pos in range(num_sentences):
        pos.append((num_sentences - 1 - 2 * min(sentence_pos, num_sentences - 1 - sentence_pos)) / (num_sentences - 1))
    return pos


def sentence_length_feature(tokenized_sentences):
    max_len = max(len(s) for s in tokenized_sentences)
    scores = [len(s) / max_len if 3 < len(s) else 0 for s in tokenized_sentences]
    return scores


def quotes_feature(sentences):
    scores = [0 if s.count('"') % 2 == 1 else 1 for s in sentences]
    return scores


def references_feature(tokenized_sentences):
    references = ['to', 'proto', 'on', 'ona', 'oni', 'jeho', 'její', 'ho', 'ji']
    scores = []
    for s in tokenized_sentences:
        score = 0
        for w in s:
            if w in references:
                score += 1
        scores.append(score)
    max_ref = max(scores)
    scores = [1 if max_ref == 0 else (max_ref - score) / max_ref for score in scores]
    return scores


def summarize(text):
    # SPLIT TO PARAGRAPHS
    pre_paragraphs = text.split('\n')
    paragraphs = []
    for i, p in enumerate(pre_paragraphs):
        if not re.match(r'^\s*$', p) and (i == len(pre_paragraphs) - 1 or re.match(r'^\s*$', pre_paragraphs[i+1])):
            paragraphs.append(p)
    # print(f'Num of paragraphs: {len(paragraphs)}')
    # for i, p in enumerate(paragraphs):
    #     print(f'par#{i+1}: {p}')

    # SPLIT TO SENTENCES
    sentences = separator.separate(text)
    print(f'Num of sentences: {len(sentences)}')
    for i, s in enumerate(sentences):
        print(f'#{i+1}: {s}')

    # TOKENIZE
    stem = False
    if stem:
        tokenized_sentences = [[czech_stemmer.cz_stem(word, aggressive=False) for word in sentence]
                               for sentence in tokenize(sentences)]
    else:
        tokenized_sentences = tokenize(sentences)

    # REMOVE STOPWORDS
    tokenized_sentences_without_stopwords = remove_stop_words(tokenized_sentences, keep_case=False)
    sentences_without_stopwords_case = remove_stop_words(sentences, keep_case=True, is_tokenized=False,
                                                         return_tokenized=False)
    print('===Sentences without stopwords===')
    for i, s in enumerate(tokenized_sentences_without_stopwords):
        print(f'''#{i+1}: {' '.join(s)}''')

    print('===Sentences without stopwords CASE===')
    for i, s in enumerate(sentences_without_stopwords_case):
        print(f'''#{i+1}: {s}''')

    # POS-TAG
    tagged_sentences = pos_tag(sentences_without_stopwords_case)
    print('=====Tagged_sentences=====')
    for i, s in enumerate(tagged_sentences):
        print(f'''#{i+1}: {s}''')

    # 1. THEMATICITY FEATURE
    thematicity_feature_scores = thematicity_feature(tokenized_sentences_without_stopwords)

    # 2. SENTENCE POSITION FEATURE - NOTE: shitty!
    sentence_position_scores = sentence_position_feature(len(sentences))

    # 3. SENTENCE LENGTH FEATURE
    sentence_length_scores = sentence_length_feature(tokenized_sentences)

    # 4. SENTENCE PARAGRAPH POSITION FEATURE

    # 5. PROPER_NOUN FEATURE
    proper_noun_scores = proper_noun_feature(tagged_sentences)

    # 6. NUMERALS FEATURE
    numerals_scores = numerals_feature(tokenized_sentences)

    # 7. NAMED ENTITIES FEATURE - very similar to PROPER_NOUN FEATURE

    # 8. TF_ISF FEATURE - NOTE: TextRank instead of TS_ISF ??? ts_isf_orig is meh
    tf_isf_scores = tf_isf_orig_feature(tokenized_sentences_without_stopwords)

    # 9. CENTROID SIMILARITY FEATURE
    centroid_similarity_scores = centroid_similarity_feature(sentences, tf_isf_scores)

    # 10. UPPER-CASE FEATURE (not in the paper)
    upper_case_scores = upper_case_feature(tokenized_sentences)

    # 11. QUOTES FEATURE (not in the paper)
    quotes_scores = quotes_feature(sentences)

    # 12. REFERENCES FEATURE (not in the paper)
    references_scores = references_feature(tokenized_sentences)

    # 13. TEXTRANK FEATURE (not in the paper)
    textrank_scores = textrank.textrank(tokenized_sentences, True, '4-1-0.0001')

    feature_matrix = []
    feature_matrix.append(thematicity_feature_scores)
    feature_matrix.append(sentence_position_scores)
    feature_matrix.append(sentence_length_scores)
    feature_matrix.append(proper_noun_scores)
    feature_matrix.append(numerals_scores)
    feature_matrix.append(tf_isf_scores)
    feature_matrix.append(centroid_similarity_scores)
    feature_matrix.append(upper_case_scores)

    features = ['  thema', 'sen_pos', 'sen_len', '  propn', '    num', ' tf_isf', 'cen_sim', '  upper']

    feature_matrix_2 = np.zeros((len(sentences), len(features)))
    for i in range(len(features)):
        for j in range(len(sentences)):
            feature_matrix_2[j][i] = feature_matrix[i][j]

    feature_sum = []
    for i in range(len(np.sum(feature_matrix_2, axis=1))):
        feature_sum.append(np.sum(feature_matrix_2, axis=1)[i])

    print('=====Scores=====')
    print(35 * ' ', end='|')
    for f in features:
        print(f, end='|')
    print()
    for i, s in enumerate(sentences):
        print(f'#{"{:2d}".format(i + 1)}: {s[:30]}', end='|')
        for f_s in feature_matrix:
            print('{: .4f}'.format(round(f_s[i], 4)), end='|')
        print('{: .4f}'.format(round(feature_sum[i], 4)))

    print('Training rbm...')
    rbm_trained = rbm.test_rbm(dataset=feature_matrix_2, learning_rate=0.1, training_epochs=14, batch_size=5,
                               n_chains=5, n_hidden=len(features))
    # another implementation of rbm, from sklearn
    # rbm2 = BernoulliRBM(n_components=len(features), n_iter=14, batch_size=5, learning_rate=0.1)
    # rbm_trained = rbm2.fit_transform(feature_matrix_2)
    # print(rbm_trained)
    rbm_trained_sums = np.sum(rbm_trained, axis=1)

    print('=====RBM Enhanced Scores=====')
    print(35 * ' ', end='|')
    for f in features:
        print(f, end='|')
    print()
    for i, s in enumerate(sentences):
        print(f'#{"{:2d}".format(i + 1)}: {s[:30]}', end='|')
        for f_s in rbm_trained[i]:
            print('{: .4f}'.format(round(f_s, 4)), end='|')
        print('{: .4f}'.format(round(rbm_trained_sums[i], 4)))

    enhanced_feature_sum = []
    feature_sum = []

    for i in range(len(np.sum(rbm_trained, axis=1))):
        enhanced_feature_sum.append([np.sum(rbm_trained, axis=1)[i], i])
        feature_sum.append([np.sum(feature_matrix_2, axis=1)[i], i])

    print(f'enhanced_feature_sum: {enhanced_feature_sum}')
    print(f'feature_sum: {feature_sum}')

    enhanced_feature_sum.sort(key=lambda x: x[0])
    feature_sum.sort(key=lambda x: -1 * x[0])
    print('=====Sorted=====')
    print(f'enhanced_feature_sum: {enhanced_feature_sum}')
    print(f'feature_sum: {feature_sum}')

    # print('=====The text=====')
    # for x in range(len(sentences)):
    #     print(sentences[x])

    extracted_sentences_rbm = []
    extracted_sentences_rbm.append([sentences[0], 0])
    extracted_sentences_simple = []
    extracted_sentences_simple.append([sentences[0], 0])

    summary_length = max(min(round(len(sentences) / 4), 12), 3)  # length between 3-12 sentences
    for x in range(summary_length):
        if enhanced_feature_sum[x][1] != 0:
            extracted_sentences_rbm.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
        if feature_sum[x][1] != 0:
            extracted_sentences_simple.append([sentences[feature_sum[x][1]], feature_sum[x][1]])

    extracted_sentences_rbm.sort(key=lambda x: x[1])
    extracted_sentences_simple.sort(key=lambda x: x[1])

    final_text_rbm = ''
    for i in range(len(extracted_sentences_rbm)):
        final_text_rbm += extracted_sentences_rbm[i][0] + '\n'
    final_text_simple = ''
    for i in range(len(extracted_sentences_simple)):
        final_text_simple += extracted_sentences_simple[i][0] + '\n'

    print('=====Extracted Final Text RBM=====')
    print(final_text_rbm)
    print()
    print('=====Extracted Final Text simple=====')
    print(final_text_simple)

    return final_text_rbm
    # return final_text_simple


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'r') as f:
            content = f.read()
            summary = summarize(content)
            print(f'===Original text===\n{content}\n')
            print(f'===Summary===\n{summary}')
    else:
        my_dir = os.path.dirname(os.path.realpath(__file__))
        article_files = os.listdir(f'{my_dir}/articles')
        total_articles = 0

        for filename in article_files:
            file_name, file_extension = os.path.splitext(filename)
            print(f'=========================Soubor: {filename}=============================')
            print('========================================================================')

            tree = ET.parse(f'{my_dir}/articles/{filename}')
            root = tree.getroot()
            articles = list(root)
            article_number = 0

            for article in articles:
                title = article.find('nadpis').text.strip()
                content = article.find('text').text.strip()
                print(f'Článek {article_number}: {title}')

                summary = summarize(content)

                output_file_name = f'{file_name}-{article_number}_system.txt'

                with open(f'{my_dir}/rouge_2_0/summarizer/system/{output_file_name}', 'w') as output_file:
                    output_file.write(summary)

                article_number += 1
                total_articles += 1
        print(f'Tested {total_articles} articles.')
        print(f'Resulting summaries stored to {my_dir}/rouge_2_0/summarizer/system/')
        print_rouge_scores(rougen=1)
        print_rouge_scores(rougen=2)


if __name__ == "__main__":
    main()
