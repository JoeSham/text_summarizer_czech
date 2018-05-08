# text_summarizer_czech

This project implements 3 different article summarizers for
the Czech language:

1. summarizer_features_rbm.py
    - based on this article: https://arxiv.org/pdf/1708.04439.pdf
    - uses a combination of several features extracted from the text
    and an RBM to enrich those features

2. summarizer_nlphackers_textrank.py
    - based mostly on this article: http://nlpforhackers.io/textrank-text-summarization/
    - uses an implementation of the TextRank algorithm from the text,
    but with Okapi BM25 similarity function

3. summarizer_machovec_modified.py
    - based on a thesis by Petr Machovec at Masaryk University: https://is.muni.cz/th/359331/fi_m/Diplomova_prace.pdf
    - uses Petr's implementation of the TextRank algorithm, with some slight improvements

### Usage
Python 3.6+ needed to run the scripts.

Install requirements:
```
pip install -r requirements.txt
```

Run one of the 3 summarizers **without any arguments** to see how
it performs on a test set of several articles.

Run one of the 3 summarizers **with 1 argument, a path to a file**
to have it print out a summary of the text in the file. The file has
to contain just simple text (no xml, ...).

For example:
```
python summarizer_nlphackers_textrank.py my_article.txt
```
