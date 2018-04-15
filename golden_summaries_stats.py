"""
Performs analysis of the golden summaries and prints out some useful statistics, like:
- avg number of sentences
- avg number of words per sentence
- frequency of a sentence_by_order in a golden summary (i.e. 1st sentence is in 559/660 summaries)
"""
import os

import xml.etree.ElementTree as ET

import separator

my_dir = os.path.dirname(os.path.realpath(__file__))
print(f'dir: {my_dir}')
article_files = os.listdir(f'{my_dir}/articles')

articles_in_sentences = {}
for filename in article_files:
    file_name, file_extension = os.path.splitext(filename)

    tree = ET.parse(f'{my_dir}/articles/{filename}')
    root = tree.getroot()
    articles = list(root)
    article_number = 0

    for article in articles:
        title = article.find('nadpis').text.strip()
        content = article.find('text').text.strip()
        # SPLIT TO SENTENCES
        sentences = separator.separate(content)
        for s in range(len(sentences)):
            sentences[s] = sentences[s].strip('" \n')
        articles_in_sentences[f'{filename.split(".")[0]}-{article_number}'] = sentences
        article_number += 1

print(articles_in_sentences)

dirr = os.path.dirname(os.path.realpath(__file__))
golden_filenames = os.listdir(f'{dirr}/rouge_2_0/summarizer/reference')

sentence_number_dict = {}
avg_words_list = []
avg_lines_list = []
prev_file_name = ''
lines = 0
summaries = 0
words = 0
for golden_filename in golden_filenames:
    file_name_original = os.path.splitext(golden_filename)[0]
    file_name, file_anot = file_name_original.split('_', 1)
    if file_name != prev_file_name and prev_file_name != '':
        avg_lines = lines / summaries
        avg_lines_list.append(avg_lines)
        avg_words = words / lines
        avg_words_list.append(avg_words)
        print(f'Soubor: {prev_file_name}')
        print(f'avg_words: {avg_words}; avg_lines: {avg_lines}; summaries: {summaries}')
        print()
        summaries = 0
        lines = 0
        words = 0
    with open(f'{dirr}/rouge_2_0/summarizer/reference/{golden_filename}', 'r') as golden_file:
        for line in golden_file:
            line = line.strip('" \n')
            lines += 1
            for word in line.split():
                words += 1
            line_index = articles_in_sentences[file_name].index(line)
            sentence_number_dict.setdefault(line_index, 0)
            sentence_number_dict[line_index] += 1
    summaries += 1
    prev_file_name = file_name
avg_lines = lines / summaries
avg_words = words / lines
avg_lines_list.append(avg_lines)
avg_words_list.append(avg_words)
print(f'Soubor: {file_name}')
print(f'avg_words: {avg_words}; avg_lines: {avg_lines}; summaries: {summaries}')
print()
avg_words_total = sum(avg_words_list) / len(avg_words_list)
print(f'avg_words_per_sentence_total: {avg_words_total}')
print()
print('How often is a sentence on position X in golden summary?')
for key in sorted(sentence_number_dict, key=sentence_number_dict.get, reverse=True):
    print(f'S#{key}: {sentence_number_dict[key]} / {len(golden_filenames)}')
