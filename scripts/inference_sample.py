"""
For testing on internal validation data (with different settings) or external dataset
"""

from __future__ import print_function

import sys
sys.path.append('/')

import argparse

import time
import torch
import os
import pandas as pd
import yaml
import pickle
from glob import glob
import numpy as np

from models.feature_extractor import get_extractor_model
from utils.heatmap_utils import identify_ckpt, initiate_attn_model
from utils.eval_utils import ClfEvaler
from utils.exp_utils import update_config, set_seeds
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from data.ThreeDimDataset import ThreeDimFeatsBag
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--config', type=str, default="config_heatmap.yaml")
args = parser.parse_args()

def search_models(fpath='.',
                  filter={'patch': '3D', 'dec': 'attn', 'enc': 'resnet50_3d', 'pretrain': '.',
                          'dec_enc_dim': 256, 'aug': 5, 'attn_latent_dim': 64}):

    terms = '*_{patch}*__enc--{enc}--{pretrain}*__dec--{dec}--{dec_enc_dim}--{attn_latent_dim}__*aug--{aug}*'.format(**filter)
    flist = glob(fpath + '/' + terms)
    flist = [fname for fname in flist if os.path.exists(os.path.join(fname, 'result.pkl'))]
    flist.sort()

    assert len(flist) > 0, "No files found for {}".format(terms)
    print("Total {} files found!".format(len(flist)))
    return flist


if __name__ == '__main__':
    conf = update_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(device=device, seed=conf['seed_exp'])

    ###################################
    # Create lists of slides to process
    ###################################
    if conf['clinical_path'] is not None:
        df = pd.read_csv(conf['clinical_path'], dtype={'patient_id': 'str'})
    else:
        raise NotImplementedError("Not implemented")

    clf_label = conf['clf_label']

    df = df.set_index('patient_id')
    le = LabelEncoder()
    le.fit(df[clf_label].unique())
    df[clf_label] = le.transform(df[clf_label])
    df = df.rename(columns={clf_label: 'event'})

    extractor = get_extractor_model(conf['encoder'])

    # Evalers
    class_names = df['event'].unique()
    class_names.sort()

    filter_dict = {'patch': conf['patch'],
                   'dec': conf['decoder'],
                   'enc': conf['encoder'],
                   'aug': conf['numOfaug'],
                   'dec_enc_dim': conf['dec_enc_dim'],
                   'pretrain': conf['pretrain'],
                   'attn_latent_dim': conf['attn_latent_dim']}
    model_list = search_models(conf['exp_dir'], filter=filter_dict)

    print('\nTotal {} models found'.format(len(model_list)))
    for idx in range(len(model_list)):
        print(os.path.basename(model_list[idx]))

    ##############################################
    # Loop through all models matching condition #
    ##############################################
    for model_idx, model_path in enumerate(model_list):
        print('\nTesting with model {}/{}'.format(model_idx+1, len(model_list)))

        # Load result
        result_path = os.path.join(model_path, 'result.pkl')
        with open(result_path, 'rb') as f:
            info = pickle.load(f)
        # Dictionary mapping between subject and which CV fold it is associated with
        if conf['mode'] == 'external':
            subject_list = df.index.values.tolist()
            fold_dict = {subj: np.arange(conf['numOffolds'], dtype=np.int8) for subj in subject_list}
        elif conf['mode'] == 'internal':
            subject_list = info['subject']
            fold_dict = {subj: fold for subj, fold in zip(subject_list, info['fold'])}
        else:
            raise NotImplementedError("Not implemented")
        print(fold_dict)
        config_exp = yaml.safe_load(open(os.path.join(model_path, 'conf.yaml'), 'r'))

        for iter_idx in range(conf['numOfiterations']):

            print("Iteration {}/{}".format(iter_idx + 1, conf['numOfiterations']))
            evaler_fold = {fold_idx: ClfEvaler(class_names=class_names, loss=conf['loss']) for fold_idx in range(conf['numOffolds'])}
            evaler_all = ClfEvaler(class_names=class_names, loss=conf['loss'])

            #######################################
            # Loop through all slides in the list #
            #######################################
            subject_list = []
            feats_list = []
            y_true_list = []
            y_pred_list = []
            for patient_id in tqdm(df.index.values):

                s = time.time()
                ########################
                # Load attention model #
                ########################

                attn_model_dict = {"dropout": config_exp['dropout'],
                                   # "out_dim": config_exp['numOfclasses'],
                                   "out_dim": 1,
                                   'attn_latent_dim': config_exp['attn_latent_dim'],
                                   'decoder': config_exp['decoder'],
                                   'decoder_enc': config_exp['decoder_enc'],
                                   'decoder_enc_dim': config_exp['decoder_enc_dim'],
                                   'context': config_exp['context'],
                                   'context_network': config_exp['context_network'] if 'context_network' in config_exp else 'GRU',
                                   'input_dim': extractor.get_output_dim()}

                ckpt_list = identify_ckpt(patient_id, fold_dict)
                if not isinstance(ckpt_list, list):
                    ckpt_list = [ckpt_list]

                for ckpt_name in ckpt_list:
                    ckpt_path = os.path.join(model_path, 'checkpoints', 'ckpt_split--{}.pt'.format(str(ckpt_name)))
                    attn_model = initiate_attn_model(model_dict=attn_model_dict,
                                                     ckpt_path=ckpt_path,
                                                     verbose=False)
                    attn_model.to(device)
                    kwargs = {'num_workers': 15, 'pin_memory': True} if device.type == "cuda" else {}
                    dataset_test = ThreeDimFeatsBag(path=conf['feats_path'],
                                                    data_df=df.loc[[patient_id]],
                                                    sample_mode='seq_num',
                                                    sample_prop=conf['numOfslices'])

                    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, **kwargs)
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(loader_test):

                            index, data, coords, y_true = batch
                            data = data.to(device)
                            y_true = y_true.to(device)
                            z, attn_dict = attn_model(data, coords)

                            evaler_fold[ckpt_name].log(index=index, z=z, y_true=y_true)
                            evaler_all.log(index=index, z=z, y_true=y_true)

                e = time.time()

            results_fold = {fold_idx: evaler_fold[fold_idx].get_metrics() for fold_idx in range(conf['numOffolds'])}
            results_all = evaler_all.get_metrics()

            # Save results
            results = {'metrics_all': results_all, 'metrics_fold': results_fold}
            print(results['metrics_all'])

            seed = os.path.basename(model_path).split('__')[0].split('--')[-1]
            patch_train = conf['patch']
            patch_test = conf['feats_path'].split('/')[-3]

            path_dict = {'seed': seed,
                         'train': patch_train,
                         'test': patch_test,
                         'encoder': conf['encoder'],
                         'decoder': conf['decoder'],
                         'numOfaug': conf['numOfaug'],
                         'numOfslices': conf['numOfslices'],
                         'dec_enc_dim': config_exp['decoder_enc_dim'],
                         'attn_latent_dim': config_exp['attn_latent_dim'],
                         'idx': iter_idx}

            basename = 'seed--{seed}__train--{train}__test--{test}_slices--{numOfslices}__enc--{encoder}'\
                        '__dec--{decoder}--{dec_enc_dim}--{attn_latent_dim}__aug--{numOfaug}__iter--{idx}'.format(**path_dict)

            result_path_new = os.path.join(conf['save_dir'], basename)
            os.makedirs(result_path_new, exist_ok=True)

            # Save config file
            with open(os.path.join(result_path_new, 'conf.yaml'), 'w') as f:
                yaml.dump(path_dict, f)

            with open(os.path.join(result_path_new, 'result.pkl'), 'wb') as f:
                pickle.dump(results, f)

