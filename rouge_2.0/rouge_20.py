""" This module executes original ROUGE 2.0 and computes average score for all candidate summaries """

import argparse
import os
import re
import subprocess

import numpy

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f'Current_dir: {cur_dir}')
    
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
    
    count = 0
    
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

'''
Results:
RBM: RECALL: 0.18250220000000003, PRECISION: 0.1780528, F_SCORE: 0.1793082
Simple: RECALL: 0.16794160000000002, PRECISION: 0.163395, F_SCORE: 0.1643356

RBM_noquotes: RECALL: 0.19485, PRECISION: 0.1890258, F_SCORE: 0.19102879999999994

RBM vs correct_format: RECALL: 0.37887119999999996, PRECISION: 0.3904378, F_SCORE: 0.380293660000000007
Simple vs correct_format: RECALL: 0.303626, PRECISION: 0.30853499999999995, F_SCORE: 0.30370759999999997

Simple with TextRank:
  RECALL: 0.4173702000000001, PRECISION: 0.4508914, F_SCORE: 0.4311302000000001  
Simple with TextRank without ts_isf and cen_sim:
  RECALL: 0.44694559999999994, PRECISION: 0.4513285999999999, F_SCORE: 0.446647
Just TextRank:
  RECALL: 0.4539810000000001, PRECISION: 0.4653058, F_SCORE: 0.45799599999999996
TextRak * 4? + SenLen + Thema:
  RECALL: 0.47882680000000005, PRECISION: 0.4697734, F_SCORE: 0.4726952
TextRank * 4 + SenLen + Numerals + UpperCase:
  RECALL: 0.4858064, PRECISION: 0.46908840000000007, F_SCORE: 0.4756082
TextRank(lcs) * 4 + SenLen + Numerals + UpperCase:
  RECALL: 0.5290358, PRECISION: 0.440043, F_SCORE: 0.47896019999999995
TextRank(lcs):
  RECALL: 0.5232748, PRECISION: 0.4500162, F_SCORE: 0.48213139999999993

8-features + original_rbm:
  RECALL: 0.42015260000000004, PRECISION: 0.42417540000000004, F_SCORE: 0.4194556

MyTextRank + BM25:
  RECALL: 0.432363, PRECISION: 0.4156085999999999, F_SCORE: 0.42212039999999995
MyTextRank + BM25 + Stemming(light):
  RECALL: 0.4253762, PRECISION: 0.40092839999999996, F_SCORE: 0.4112562000000001
MyTextRank + BM25 + 1st sentence extra:
  RECALL: 0.46459880000000003, PRECISION: 0.4245344, F_SCORE: 0.4412731999999999
MyTextRank + BM25 + always 1st sentence:
  RECALL: 0.4512107692307692, PRECISION: 0.4408628846153847, F_SCORE: 0.44436942307692295

After taking round(25%) top sentences instead of floor(25%):
   
MyTextRank + BM25 + always 1st sentence, d=0.85
  RECALL: 0.4512107692307692, PRECISION: 0.4408628846153847, F_SCORE: 0.44436942307692295
MyTextRank + BM25 + always 1st sentence, d=0.95  
  RECALL: 0.46471461538461545, PRECISION: 0.4362459615384615, F_SCORE: 0.448815
MyTextRank + BM25 + always 1st sentence, d=0.9    
  RECALL: 0.46477192307692317, PRECISION: 0.4363728846153847, F_SCORE: 0.44893192307692303

8-features + original_rbm:  
  RECALL: 0.4396255769230769, PRECISION: 0.424495, F_SCORE: 0.4299136538461539

TextRankMachovec, 3-1-inf (lcs):  
  RECALL: 0.5282530769230769, PRECISION: 0.44429519230769227, F_SCORE: 0.4811921153846153
TextRankMachovec, 4-1-inf (bm25):  
  RECALL: 0.5274344230769231, PRECISION: 0.44312903846153845, F_SCORE: 0.4802309615384614
TextRankMachovec, 2-1-inf (cosine):  
  RECALL: 0.46998038461538455, PRECISION: 0.4657380769230769, F_SCORE: 0.466485
  
'''
