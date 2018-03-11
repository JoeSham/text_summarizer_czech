import collections
import logging
import math
import os
import re
import sys

import numpy as np

import joe.rbm
import joe.separator

from RDRPOSTagger_python_3.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from RDRPOSTagger_python_3.Utility.Utils import readDictionary
os.chdir('../..')  # because above modules do chdir ... :/

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


# def remove_stop_words(sentences, keep_case=False):
#     sentences_without_stopwords = []
#     for sentence_orig in sentences:
#         sentence_without_stopwords = []
#         if keep_case:
#             words = sentence_orig.split()
#         else:
#             words = sentence_orig.lower().split()
#         for word in words:
#             if word.lower() not in STOPWORDS:
#                 sentence_without_stopwords.append(word)
#         sentences_without_stopwords.append(' '.join(sentence_without_stopwords))
#     return sentences_without_stopwords

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
        tokenized.append(s.split())
    return tokenized


def thematicity_feature(tokenized_sentences, most_common_cutoff=10):
    # alnum_words = []
    # for sentence in tokenized_sentences:
    #     for word in sentence:
    #         is_alnum = True
    #         for c in word:
    #             if not c.isalnum():
    #                 is_alnum = False
    #                 break
    #         if is_alnum:
    #             alnum_words.append(word)
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
        thematicity = count_of_thematic_words / len(sentence)
        thematicity_scores.append(thematicity)
    return thematicity_scores


def upper_case_feature(tokenized_sentences):
    tokenized_sentences_wo_sw = remove_stop_words(tokenized_sentences, keep_case=True)
    scores = []
    for sentence in tokenized_sentences_wo_sw:
        count_of_uppercase_starting_words = 0
        for word in sentence:
            if word[0].isupper():
                count_of_uppercase_starting_words += 1
        scores.append(count_of_uppercase_starting_words / len(sentence))
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
        scores.append(score / len(sentence))
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
#         scores.append(tf_isf / len(sentence))
#     return scores


def proper_noun_feature(tagged):
    scores = []
    for sentence in tagged:
        score = 0
        for word, tag in sentence:
            if tag == 'PROPN':
                score += 1
        scores.append(score / len(sentence))
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
        scores.append(score / len(sentence))
    return scores


def sentence_position_feature(num_sentences):
    threshold = 0.2 * num_sentences
    min_v = threshold * num_sentences
    max_v = threshold * 2 * num_sentences
    pos = []
    for sentence_pos in range(num_sentences):
        if sentence_pos in (0, num_sentences - 1):
            pos.append(1)
        else:
            t = math.cos((sentence_pos - min_v) * ((1 / max_v) - min_v))
            pos.append(t)
    return pos


def sentence_length_feature(tokenized_sentences):
    max_len = max(len(s) for s in tokenized_sentences)
    scores = [len(s) / max_len if 3 < len(s) < 11 else 0 for s in tokenized_sentences]
    return scores


def summarize(text):
    # SPLIT TO PARAGRAPHS
    pre_paragraphs = text.split('\n')
    paragraphs = []
    for i, p in enumerate(pre_paragraphs):
        if not re.match(r'^\s*$', p) and (i == len(pre_paragraphs) - 1 or re.match(r'^\s*$', pre_paragraphs[i+1])):
            paragraphs.append(p)
    print(f'Num of paragraphs: {len(paragraphs)}')
    for i, p in enumerate(paragraphs):
        print(f'par#{i+1}: {p}')

    # SPLIT TO SENTENCES
    sentences = joe.separator.separate(text)
    print(f'Num of sentences: {len(sentences)}')
    for i, s in enumerate(sentences):
        print(f'#{i+1}: {s}')

    # TOKENIZE
    tokenized_sentences = tokenize(sentences)

    # REMOVE STOPWORDS
    tokenized_sentences_without_stopwords = remove_stop_words(tokenized_sentences, keep_case=False)
    sentences_without_stopwords_case = remove_stop_words(sentences, keep_case=True, is_tokenized=False,
                                                         return_tokenized=False)
    print('===Sentences without stopwords===')
    for i, s in enumerate(tokenized_sentences_without_stopwords):
        print(f'''#{i+1}: {' '.join(s)}''')

    # POS-TAG
    tagged_sentences = pos_tag(sentences_without_stopwords_case)
    print(f'tagged_sentences: {tagged_sentences}')

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

    feature_matrix = []
    feature_matrix.append(thematicity_feature_scores)
    feature_matrix.append(sentence_position_scores)
    feature_matrix.append(sentence_length_scores)
    feature_matrix.append(proper_noun_scores)
    feature_matrix.append(numerals_scores)
    feature_matrix.append(tf_isf_scores)
    feature_matrix.append(centroid_similarity_scores)
    feature_matrix.append(upper_case_scores)

    print('=====Scores=====')
    features = ['  thema', 'sen_pos', 'sen_len', '  propn', '    num', ' tf_isf', 'cen_sim', '  upper']
    print(35 * ' ', end='|')
    for f in features:
        print(f, end='|')
    print()
    for i, s in enumerate(sentences):
        print(f'#{"{:2d}".format(i + 1)}: {s[:30]}', end='|')
        for f_s in feature_matrix:
            print('{: .4f}'.format(round(f_s[i], 4)), end='|')
        print()

    feature_matrix_2 = np.zeros((len(sentences), 8))
    for i in range(8):
        for j in range(len(sentences)):
            feature_matrix_2[j][i] = feature_matrix[i][j]

    # print("\n\n\nPrinting Feature Matrix Normed: ")
    # feature_matrix_normed = feature_matrix / feature_matrix.max(axis=0)
    # feature_matrix_normed = feature_matrix

    feature_sum = []

    for i in range(len(np.sum(feature_matrix_2, axis=1))):
        feature_sum.append(np.sum(feature_matrix_2, axis=1)[i])

    print('Training rbm...')
    rbm_trained = joe.rbm.test_rbm(dataset=feature_matrix_2, learning_rate=0.1, training_epochs=14, batch_size=5,
                                   n_chains=5, n_hidden=8)
    print('Training rbm done')
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

    enhanced_feature_sum.sort(key=lambda x: -1 * x[0])
    feature_sum.sort(key=lambda x: x[0])
    print('=====Sorted=====')
    print(f'enhanced_feature_sum: {enhanced_feature_sum}')
    print(f'feature_sum: {feature_sum}')

    print('=====The text=====')
    for x in range(len(sentences)):
        print(sentences[x])

    extracted_sentences = []
    extracted_sentences.append([sentences[0], 0])
    extracted_sentences_2 = []
    extracted_sentences_2.append([sentences[0], 0])

    # length_to_be_extracted = len(enhanced_feature_sum) // 2
    summary_length = max(min(len(sentences) // 4, 8), 3)  # length between 3-8 sentences
    for x in range(summary_length):
        if enhanced_feature_sum[x][1] != 0:
            extracted_sentences.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
        if feature_sum[x][1] != 0:
            extracted_sentences_2.append([sentences[feature_sum[x][1]], feature_sum[x][1]])

    extracted_sentences.sort(key=lambda x: x[1])
    extracted_sentences_2.sort(key=lambda x: x[1])

    final_text = ''
    for i in range(len(extracted_sentences)):
        final_text += extracted_sentences[i][0] + '\n'
    final_text_2 = ''
    for i in range(len(extracted_sentences_2)):
        final_text_2 += extracted_sentences_2[i][0] + '\n'

    print('=====Extracted Final Text=====')
    print(final_text)
    print()
    print('=====Extracted Final Text 2=====')
    print(final_text_2)


text1 = '''V USA demonstrovaly tisíce lidí proti policejnímu násilí

Tisíce lidí demonstrovaly v noci na čtvrtek v Baltimoru, New Yorku, Washingtonu, Bostonu a dalších amerických městech na protest proti policejní brutalitě. Protesty se obešly bez většího násilí, ale v New Yorku, kde se sešlo několik set demonstrantů, podle BBC zatkla policie šedesát lidí.

Demonstranti protestovali proti policejnímu násilí, protože v Baltimoru zemřel černošský mladík Freddie Gray na následky zranění při zatýkání. Od pondělního pohřbu lidé v Baltimoru masově protestují, ve městě platí zákaz nočního vycházení a nasazena byla Národní garda.    
    
„Bez spravedlnosti nebude mír,” skandovali ve středu večer v Baltimoru demonstranti. „Pošlete ty zabijácké policisty do vězení. Vinen je celý tento mizerný systém,” prohlašovali studenti a další mladí lidé. „Vraždící policisté si zasluhují celu,” stálo na jednom z početných transparentů.

Silné byly protesty i v New Yorku, kde lidé odsuzovali policejní násilí na černošských obyvatelích a hájili právo na protest.

Ve Spojených státech se v posledních měsících stalo několik případů, kdy bělošští policisté při zásahu zabili černocha. Baltimore se však od ostatních incidentů odlišuje, neboť má černošského policejního náčelníka i černošskou starostku. Pětadvacetiletý Gray zemřel 19. dubna na následky zlomeniny krčních obratlů, kterou utrpěl o týden dříve při zatýkání. Policie následně připustila, že policisté v rozporu s předpisy nezajistili Grayovi náležitou zdravotní péči. Když ho nakládali 12. dubna do antonu, Gray normálně komunikoval. O tři čtvrtě hodiny později už žádali o zásah zdravotníků, kteří ho převezli do nemocnice, kde po týdnu zemřel.
    
Výsledky policejního vyšetřování měly být podle dřívějších zpráv médií oznámeny 1. května. Baltimorská policie ale ve středu podle Reuters uvedla, že žádnou zprávu v pátek nezveřejní, nález předá státnímu návladnímu. Šest baltimorských policistů bylo postaveno mimo službu.
'''

summarize(text1)
