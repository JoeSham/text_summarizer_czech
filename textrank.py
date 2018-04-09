#!/usr/bin/env python

"""
This module includes 'textrank' function, which performs TextRank algorithm with given lists of sentences
and words, i.e. it detects the salience of sentences. Indexes of sentences ordered descendingly by their computed
scores are returned in a list.
"""

import math
import scipy.spatial.distance as scp
import networkx


def textrank(sentences, use_words, config):
    config_values = config.split("-")
    if len(config_values) != 3:
        raise ValueError("Nesprávný počet položek konfiguračního kódu metody Textrank, musí být přesně 3")
    
    try:
        sim_method = int(config_values[0])
        d = float(config_values[1])
        stop_threshold = float(config_values[2])
    except ValueError:
        raise ValueError("Nesprávná hodnota některé položky konfiguračního kódu metody Textrank, první položka musí"
                         "být celé číslo, druhá a třetí položka musí být reálná čísla")
    
    all_ok = True
    message = ""

    if sim_method not in range(1, 5):
        message += "Způsob porovnávání podobnosti vět může mít pouze hodnotu od 1 do 4"
        all_ok = False
    
    if d < 0 or d > 1:
        message += "; Faktor tlumení (damping factor) může mít pouze hodnotu od 0 do 1"
        all_ok = False
    
    if stop_threshold <= 0:
        message += "; Práh zastavení musí být větší než 0"
        all_ok = False
    
    if not all_ok:
        message = message.strip(";")
        message = message.strip()
        raise ValueError(message)
        
    init_score = 1 / float(len(sentences))
    sim = 0
    sim_sum = 0

    all_words = set([word for s in sentences for word in s])
    avg_idf = calc_avg_idf(sentences, all_words)
    avg_len = avg_sentence_length(sentences)
    
    graph = networkx.Graph()  # Creating an undirected graph
    for i in range(0, len(sentences)):
        for j in range(0, len(sentences)):
            graph.add_node(i, score=init_score)
            graph.add_node(j, score=init_score)
            
            if sim_method == 1:
                sim = similarity(sentences[i], sentences[j], use_words)
            elif sim_method == 2:
                sim = cossim(sentences[i], sentences[j], use_words)
            elif sim_method == 3:
                sim = lcs(sentences[i], sentences[j], use_words)
            else:
                sim = bm25(sentences[i], sentences[j], sentences, avg_idf, avg_len)
                
            graph.add_edge(i, j, value=sim)
            sim_sum += sim
    
    go_on =  True
    new_score = 0
    partial = 0
    
    # Variable 'sim' is used again for another purpose
    while go_on:
        for i in range(0, len(sentences)):
            partial = 0
            for j in range(0, len(sentences)):
                sim = graph.get_edge_data(i,j)["value"]
                partial += (sim / float(sim_sum)) * dict(graph.nodes(True))[j]["score"]
            
            new_score = (1 - d) + (d * partial)
            actual_score = dict(graph.nodes(True))[i]["score"]
            
            graph.add_node(i, score=new_score)
            
            if abs(actual_score - new_score) < stop_threshold:
                go_on = False
                
    score_dict = dict()
    for i in range(0, len(sentences)):
        score_dict[i] = dict(graph.nodes(True))[i]["score"]

    index_list = sorted(score_dict, key=score_dict.__getitem__)
    index_list.reverse()

    return index_list
    # scores = []
    # for i in range(0, len(sentences)):
    #     scores.append(dict(graph.nodes(True))[i]["score"])
    # max_score = max(scores)
    # scores = [score / max_score for score in scores]
    # return scores


def similarity(s1, s2, use_words):
    if not use_words:
        w1 = set(s1.lemmas)
        w2 = set(s2.lemmas)
    else:
        w1 = set(s1)
        w2 = set(s2)
    
    total = len(w1 & w2) / float(1 + math.log10(len(w1)) + math.log10(len(w2)))
    
    return total


def cossim(s1, s2, use_words):
    if not use_words:
        w = list(set(s1.lemmas) | set(s2.lemmas))
    else:
        w = list(set(s1) | set(s2))
    
    vec1 = list()
    vec2 = list()
    
    for word in w:
        if not use_words:
            vec1.append(s1.lemmas.count(word))
            vec2.append(s2.lemmas.count(word))
        else:
            vec1.append(s1.count(word))
            vec2.append(s2.count(word))
    
    distance = scp.cosine(vec1, vec2)
    return 1 - distance


def lcs(s1, s2, use_words):
    if not use_words:
        x = s1.lemmas
        y = s2.lemmas
    else:
        x = s1
        y = s2
         
    # Taken from http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    n = len(x)
    m = len(y)
    table = dict()
 
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
            else:
                table[i, j] = max(table[i-1, j], table[i, j-1])
                
    return table[n, m]


def frequency_in_sentence(term, tokenized_sentence):
    freq = 0
    term = term.lower()
    for w in tokenized_sentence:
        if term == w.lower():
            freq += 1
    return freq


def avg_sentence_length(tokenized_sentences):
    return sum([len(s) for s in tokenized_sentences]) / max(len(tokenized_sentences), 1)


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
        return math.log((num_sentences - sentences_with_term + 0.5)) - math.log(sentences_with_term + 0.5)
    else:
        if sentences_with_term <= num_sentences / 2:
            return math.log((num_sentences - sentences_with_term + 0.5)) - math.log(sentences_with_term + 0.5)
        else:
            return eps * avg_idf


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
