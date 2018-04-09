import collections
import logging
import math
import os
from operator import itemgetter
import re
import sys

import numpy as np
import xml.etree.ElementTree as ET

import czech_stemmer
from RDRPOSTagger_python_3.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from RDRPOSTagger_python_3.Utility.Utils import readDictionary
os.chdir('../..')  # because above modules do chdir ... :/
import separator

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


def pagerank(adjacency_matrix, eps=0.0001, d=0.9):
    p = np.ones(len(adjacency_matrix)) / len(adjacency_matrix)
    while True:
        new_p = np.ones(len(adjacency_matrix)) * (1 - d) / len(adjacency_matrix) + d * adjacency_matrix.T.dot(p)
        delta = abs((new_p - p).sum())
        if delta <= eps:
            return new_p
        p = new_p


def idf(term, tokenized_sentences, avg_idf=None, eps=0.25):
    term = term.lower()
    sentences_with_term = 0
    for sentence in tokenized_sentences:
        for word in sentence:
            if term == word.lower():
                sentences_with_term += 1
                break
    num_sentences = len(tokenized_sentences)
    if avg_idf is None:
        idf_score = math.log((num_sentences - sentences_with_term + 0.5)) - math.log(sentences_with_term + 0.5)
    else:
        if sentences_with_term <= num_sentences / 2:
            idf_score = math.log((num_sentences - sentences_with_term + 0.5)) - math.log(sentences_with_term + 0.5)
        else:
            idf_score = eps * avg_idf
    return idf_score


def frequency_in_sentence(term, tokenized_sentence):
    freq = 0
    term = term.lower()
    for w in tokenized_sentence:
        if term == w.lower():
            freq += 1
    return freq


def avg_sentence_length(tokenized_sentences):
    return sum([len(s) for s in tokenized_sentences]) / max(len(tokenized_sentences), 1)


def calc_avg_idf(tokenized_sentences, all_words):
    sum_idf = 0
    for word in all_words:
        sum_idf += idf(word, tokenized_sentences)
    return sum_idf / len(all_words)


def bm25(s1, s2, tokenized_sentences, avg_idf, avg_len, k1=1.2, b=0.75):
    score = 0
    for word in s2:
        fq = frequency_in_sentence(word, s1)
        score += idf(word, tokenized_sentences, avg_idf) * fq * (k1 + 1) / (fq + k1 * (1 - b + b * len(s1) / avg_len))
    return score


def build_similarity_matrix(tokenized_sentences, stopwords=None):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(tokenized_sentences), len(tokenized_sentences)))

    all_words = set([word for s in tokenized_sentences for word in s])
    avg_idf = calc_avg_idf(tokenized_sentences, all_words)
    avg_len = avg_sentence_length(tokenized_sentences)

    for idx1 in range(len(tokenized_sentences)):
        for idx2 in range(len(tokenized_sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = bm25(tokenized_sentences[idx1], tokenized_sentences[idx2],
                                                 tokenized_sentences, avg_idf, avg_len)

    # normalize the matrix row-wise
    for idx in range(len(similarity_matrix)):
        similarity_matrix[idx] /= max(similarity_matrix[idx].sum(), 1)

    return similarity_matrix


def textrank(tokenized_sentences, top_n=5, stopwords=None):
    """
    tokenized_sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top_n = how may sentences the summary should contain
    stopwords = a list of stopwords
    """
    similarity_matrix = build_similarity_matrix(tokenized_sentences, stopwords)
    sentence_ranks = pagerank(similarity_matrix)

    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    return ranked_sentence_indexes
    # sorted_sentence_indexes = sorted(ranked_sentence_indexes[:top_n])
    # summary = itemgetter(*selected_sentences)(tokenized_sentences)
    # return summary


def summarize(text):
    # SPLIT TO PARAGRAPHS
    pre_paragraphs = text.split('\n')
    paragraphs = []
    for i, p in enumerate(pre_paragraphs):
        if not re.match(r'^\s*$', p) and (i == len(pre_paragraphs) - 1 or re.match(r'^\s*$', pre_paragraphs[i+1])):
            paragraphs.append(p)

    # SPLIT TO SENTENCES
    sentences = separator.separate(text)
    print(f'Num of sentences: {len(sentences)}')
    for i, s in enumerate(sentences):
        print(f'#{i+1}: {s}')

    # TOKENIZE
    stem = False
    if stem:
        tokenized_sentences = [[czech_stemmer.cz_stem(word, aggressive=True) for word in sentence]
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

    summary = ''
    counter = 0
    summary_length = max(min(round(len(sentences) / 4), 15), 3)  # length between 3-15 sentences
    ranked_sentence_indexes = textrank(tokenized_sentences_without_stopwords, stopwords=[], top_n=summary_length)
    print(f'ranked_sentence_indexes: {ranked_sentence_indexes}')
    # add 1st sentence always
    summary += f'{sentences[0]}\n'
    counter += 1
    ranked_sentence_indexes.remove(0)
    # add also 2nd sentence if it is in top50%
    if 1 in ranked_sentence_indexes[:len(ranked_sentence_indexes) // 2]:
        summary += f'{sentences[1]}\n'
        counter += 1
        ranked_sentence_indexes.remove(1)
    for sentence_index in sorted(ranked_sentence_indexes[:summary_length - counter]):
        if counter == summary_length:
            break
        summary += f'{sentences[sentence_index]}\n'
        counter += 1
    return summary


def main():
    my_dir = os.path.dirname(os.path.realpath(__file__))
    print(f'dir: {my_dir}')
    article_files = os.listdir(f'{my_dir}/articles')

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

            # if not os.path.exists(f'{dir}/test_summaries/'):
            #     os.makedirs(f'{dir}/test_summaries/')

            with open(f'{my_dir}/rouge_2.0/summarizer/system/{output_file_name}', 'w') as output_file:
                output_file.write(summary)

            article_number += 1


if __name__ == "__main__":
    main()
