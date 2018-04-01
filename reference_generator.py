#!/usr/bin/env python
# encoding=utf-8 (pep 0263)

import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--format",
                        help="form of sentences in result, 1 for original texts, 2 for words, 3 for lemmatised words (default)",
                        type=int, default=3)
    parser.add_argument("-l", "--lemmatise", help="compute lemmas in sentences", action="store_false")
    args = parser.parse_args()

    if not args.format in range(1, 4):
        sys.stderr.write(
            "Typ formátu vět má povolené pouze hodnoty 1 (původní texty), 2 (slova) a 3 (lemmatizovaná slova)\n")
        exit(1)

    dirr = os.path.dirname(os.path.realpath(__file__))

    print('TVORBA REFERENČNÍCH SOUHRNŮ PRO ROUGE 2.0')
    print()
    dir_names = os.listdir(dirr + "/reference_sources")

    for dir_name in dir_names:
        print(f'Složka: {dir_name}')
        names = os.listdir(f'{dirr}/reference_sources/{dir_name}')

        for name in names:
            file_name_original = os.path.splitext(name)[0]
            file_name, file_anot = file_name_original.split(".", 1)
            print(f'Soubor: {name}')

            reference = ''

            with open(dirr + "/reference_sources/" + dir_name + "/" + name, "r") as gold_file:
                for line in gold_file:
                    line = line.strip().replace(u"\ufeff", "")
                    reference += f'{line}\n'
                    # sentence = Sentence(line, args.lemmatise, False) #When 'args.lemmatise' is True, it works only in the FI net
                    # if args.format == 1:
                    #     reference += sentence.text + "\n"
                    # elif args.format == 2:
                    #     reference += sentence.word_text
                    # else:
                    #     reference += sentence.lemma_text + "\n"

                reference = reference.strip()

            with open(f'{dirr}/rouge_2.0/summarizer/reference/{file_name}_{file_anot}.txt', 'w') as output_file:
                output_file.write(reference)
        print()
