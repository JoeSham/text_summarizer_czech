"""
A slightly modified version of a script originally written by Petr Machovec
"""

import os

if __name__ == '__main__':
    dirr = os.path.dirname(os.path.realpath(__file__))
    print('TVORBA REFERENČNÍCH SOUHRNŮ PRO ROUGE 2.0')
    dir_names = os.listdir(f'{dirr}/golden_summaries')

    for dir_name in dir_names:
        print(f'Složka: {dir_name}')
        names = os.listdir(f'{dirr}/golden_summaries/{dir_name}')

        for name in names:
            file_name_original = os.path.splitext(name)[0]
            file_name, file_anot = file_name_original.split('.', 1)
            print(f'Soubor: {name}')
            reference = ''
            with open(f'{dirr}/golden_summaries/{dir_name}/{name}', 'r') as gold_file:
                for line in gold_file:
                    line = line.strip().replace(u'\ufeff', '')
                    reference += f'{line}\n'
                reference = reference.strip()

            with open(f'{dirr}/rouge_2_0/summarizer/reference/{file_name}_{file_anot}.txt', 'w') as output_file:
                output_file.write(reference)
        print()
