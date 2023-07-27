"""
Aggregating MIL results
"""

import os
import pickle
import argparse
import yaml

import sys
sys.path.append('../')
import glob
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Aggregation")
parser.add_argument('--mode', type=str, choices=['self', 'cross'], default='self')
parser.add_argument('--folder_path', type=str, help="Basename should be experiment setting e.g. /home/andrew/workspace/results/MIL/clf_loo")
args = parser.parse_args()

assert args.folder_path is not None, "Specify experiment path!"

####################################
# Create setup for each experiment #
####################################
if args.folder_path[-1] == '/':
    folders_list = glob.glob(args.folder_path + '*')
    task = args.folder_path.split('/')[-2].split('__')[0].split('--')[0]
else:
    folders_list = glob.glob(args.folder_path + '/*')
    task = os.path.basename(args.folder_path).split('__')[0].split('--')[0]

#########################
# Aggregate the results #
#########################

if args.mode == 'self':
    # Straightforward from saved config files
    basic_keys = ['seed_data', 'seed_exp', 'sample_prop', 'sample_mode', 'decoder', 'attn_latent_dim', 'encoder',
            'lr', 'epochs', 'prop_train', 'es', 'decoder_enc_dim', 'decoder_enc_num', 'context', 'context_network',
            'numOfaug', 'dropout', 'grad_accum', 'weight_decay', 'scheduler', 'epochs_finetune', 'lr_finetune']
    # Need additional parsing to get this information
    custom_keys = ['patch', 'pretrain']
else:
    basic_keys = ['seed', 'train', 'test', 'encoder', 'decoder', 'numOfaug']
    custom_keys = []

eval_keys = []

if task == 'clf':
    metrics_dict = {'acc': [], 'bal_acc': [], 'f1': [], 'auc': []}
    eval_keys.extend(['acc', 'bal_acc', 'f1', 'auc',
                 'bal_acc_fold',
                 'acc_fold',
                 'f1_fold',
                 'auc_fold',
                 'bal_acc_fold_avg',
                 'acc_fold_avg',
                 'f1_fold_avg',
                 'auc_fold_avg',
                 'bal_acc_fold_std',
                 'acc_fold_std',
                 'f1_fold_std',
                 'auc_fold_std',
                 ])
elif task == 'surv':
    metrics_dict = {'c_index': []}
    eval_keys.extend(['c_index', 'c_index_fold', 'c_index_fold_avg', 'c_index_fold_std'])
else:
    raise NotImplementedError("Not implemented for ", task)

pd_dict = {**{key: [] for key in basic_keys}, **{key: [] for key in custom_keys}, **{key: [] for key in eval_keys}}

counter = 0
for idx, fpath in enumerate(folders_list):
    if not os.path.isdir(fpath):
        continue

    result_path = os.path.join(fpath, 'result.pkl')
    config_path = os.path.join(fpath, 'conf.yaml')

    with open(config_path, 'r') as f:
        info = yaml.safe_load(f)

    if os.path.isfile(result_path):
        counter += 1

        ## custom keys
        tokens = os.path.basename(fpath).split('__')    # Parse from folder name
        if len(custom_keys) > 0:
            for token in tokens:
                if 'patch_' in token and 'features' not in token:
                    pd_dict['patch'].append('_'.join(token.split('_')[:-1]))

            pretrain = info['feats_path'].split('/')[-1]
            pd_dict['pretrain'].append(pretrain)

        ## basic keys
        for key in basic_keys:
            if key in info.keys():
                pd_dict[key].append(info[key])
            else:
                pd_dict[key].append(-1)

        ## eval_keys
        with open(result_path, 'rb') as f:
            info = pickle.load(f)

        for key in info['metrics_all'].keys():
            if key in eval_keys:
                pd_dict[key].extend(np.around([info['metrics_all'][key]], 3))

        if task == 'clf':
            metrics_dict = {'acc': [], 'bal_acc': [], 'f1': [], 'auc': []}
        elif task == 'surv':
            metrics_dict = {'c_index': []}

        for fold, val in info['metrics_fold'].items():
            for key in metrics_dict.keys():
                metrics_dict[key].append(val[key])

        for key, val in metrics_dict.items():
            pd_dict['{}_fold'.format(key)].append(np.around(val, 3))
            pd_dict['{}_fold_avg'.format(key)].append(np.around(np.mean(val), 3))
            pd_dict['{}_fold_std'.format(key)].append(np.around(np.std(val), 3))

print("\n Aggregated results for {} experiments!".format(counter))

df_agg = pd.DataFrame.from_dict(pd_dict)
df_agg.to_csv(os.path.join(args.folder_path, 'result_agg.csv'))