"""
This module executes original ROUGE 2.0 and computes average score for all candidate summaries
Slightly modified version of a script originally written by Petr Machovec
"""

import argparse
import os
import re
import subprocess

import numpy


def print_rouge_scores():
    print('===Rouge 2.0 evaluation===')
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--rougen', type=int, default=0)
    args = parser.parse_args()

    if args.rougen == 1 or args.rougen == 0:
        print('using 1-grams')
        rouge = subprocess.Popen(
            args='java -jar -Drouge.prop="rouge.properties1" rouge2.0.jar', shell=True, stdout=subprocess.PIPE
        )
    elif args.rougen == 2:
        rouge = subprocess.Popen(
            args='java -jar -Drouge.prop="rouge.properties2" rouge2.0.jar', shell=True, stdout=subprocess.PIPE
        )
    else:
        rouge = None
        print('Špatná velikost n-gramů')
        exit(1)

    rouge.wait()
    rouge_results = str(rouge.communicate()).split('========Results=======')[1].split('======Results End======')[0]
    results_list = rouge_results.encode('utf8').decode('unicode_escape').strip().split('\n')
    print(results_list)

    r_val_list = list()
    p_val_list = list()
    f_val_list = list()

    for result in results_list:
        r_list = list(re.findall('Average_R:[0-9].[0-9][0-9][0-9][0-9][0-9]', result)[0][10:])
        p_list = list(re.findall('Average_P:[0-9].[0-9][0-9][0-9][0-9][0-9]', result)[0][10:])
        f_list = list(re.findall('Average_F:[0-9].[0-9][0-9][0-9][0-9][0-9]', result)[0][10:])

        r_list[1] = '.'
        p_list[1] = '.'
        f_list[1] = '.'

        r_string = "".join(r_list)
        p_string = "".join(p_list)
        f_string = "".join(f_list)

        r_val_list.append(float(r_string))
        p_val_list.append(float(p_string))
        f_val_list.append(float(f_string))

    r_avg = numpy.mean(r_val_list)
    p_avg = numpy.mean(p_val_list)
    f_avg = numpy.mean(f_val_list)

    print(f'RECALL: {r_avg}, PRECISION: {p_avg}, F_SCORE: {f_avg}')


if __name__ == '__main__':
    print_rouge_scores()
