import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ckpt = 'bert-base-cased'

dev = False


semi_aug_ratio_range = range(1, 6)
cf_aug_ratio_range = range(1, 9)

for eval_task in ['conll2003', 'tech_news', 'ai', 'literature', 'music', 'politics', 'science']:
    if dev and eval_task != 'conll2003':
        continue
    ori_f1_list = []
    semi_f1_list = []
    cf_f1_list = []
    mix_f1_list = []
    best = 0
    best_idx = 0
    idx = 0
    for semi_aug_ratio in semi_aug_ratio_range:
        for cf_aug_ratio in cf_aug_ratio_range:
            if dev:
                in_domain_path = f'results/{ckpt}/semi_{semi_aug_ratio}/cf_{cf_aug_ratio}/{eval_task}_dev.csv'
            else:
                in_domain_path = f'results/{ckpt}/semi_{semi_aug_ratio}/cf_{cf_aug_ratio}/{eval_task}_test.csv'
            result = pd.read_csv(in_domain_path)

            ori_f1_list.append(result['ori_f1'].values[-1])
            semi_f1_list.append(result['semi_f1'].values[-1])
            cf_f1_list.append(result['cf_f1'].values[-1])
            mix_f1_list.append(result['mix_f1'].values[-1])
            # print(ori_f1_list[-1], semi_f1_list[-1], cf_f1_list[-1], mix_f1_list[-1])
            if best < result['mix_f1'].values[-1]:
                best = result['mix_f1'].values[-1]
                best_idx = idx
                print(semi_aug_ratio, cf_aug_ratio)
            idx += 1

    length = 4*8+1
    print(best_idx)
    # plt.plot(range(1,length), ori_f1_list, label='ori')
    # plt.plot(range(1,length), semi_f1_list, label='semi')
    # plt.plot(range(1,length), cf_f1_list, label='cf')
    # plt.plot(range(1,length), mix_f1_list, label='mix')
    # plt.legend()
    # plt.title(f'f1 trends of {ckpt}')
    # if dev:
    #     plt.savefig(f'f1_{ckpt}_dev.png')
    # else:
    #     plt.savefig(f'f1_{ckpt}_test.png')
    # plt.close()
    
    print(eval_task, ori_f1_list[best_idx], cf_f1_list[best_idx], semi_f1_list[best_idx], mix_f1_list[best_idx])
