import logging
import os
import re
import sys

import xml.etree.ElementTree as ET

import czech_stemmer
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
        # tokenized.append(s.split())
    return tokenized


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

    summary = ''
    counter = 0
    summary_length = max(min(round(len(sentences) / 4), 15), 3)  # length between 3-15 sentences
    ranked_sentence_indexes = textrank.textrank(tokenized_sentences, True, '4-1-1')
    print(f'ranked_sentence_indexes: {ranked_sentence_indexes}')
    # add 1st sentence always
    summary += f'{sentences[0]}\n'
    counter += 1
    ranked_sentence_indexes.remove(0)
    # # add also 2nd sentence if it is in top 50%
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
        print_rouge_scores()


if __name__ == "__main__":
    main()
