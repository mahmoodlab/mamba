"""
For testing variability
"""

from __future__ import print_function

import sys
sys.path.append('/')

import argparse

import torch
import os
import pandas as pd
import yaml
import pickle
from glob import glob
import numpy as np
from scipy.special import softmax

from models.feature_extractor import get_extractor_model
from utils.heatmap_utils import identify_ckpt, initiate_attn_model, normalize_ig_scores, \
                                encode_features, attend_features
from utils.exp_utils import update_config, set_seeds
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from data.ThreeDimDataset import ThreeDimFeatsBag
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--config', type=str, default="config_heatmap.yaml")
args = parser.parse_args()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## Integrated gradient
def interpret_patient(features):
    return attn_model.captum(x=features)

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

        config_exp = yaml.safe_load(open(os.path.join(model_path, 'conf.yaml'), 'r'))

        #######################################
        # Loop through all slides in the list #
        #######################################
        subject_list = []
        prob_list_all = []
        prob_cohort = []
        ig_list_all = []
        coords_list = []
        features_agg_all = []

        for patient_id in tqdm(df.index.values):
            subject_list.append(patient_id)
            ########################
            # Load attention model #
            ########################
            attn_model_dict = {"dropout": config_exp['dropout'],
                               # "out_dim": config_exp['numOfclasses'],
                               "out_dim":1,
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

                kwargs = {'num_workers': 20, 'pin_memory': True} if device.type == "cuda" else {}
                dataset_test = ThreeDimFeatsBag(path=conf['feats_path'],
                                                data_df=df.loc[[patient_id]])
                loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, **kwargs)
                print("\nPATIENT ID", patient_id)

                with torch.no_grad():
                    for batch_idx, batch in enumerate(loader_test):
                        index, data, coords, y_true = batch
                        y_true = y_true.to(device)

                        logits, _ = attn_model(data.to(device), coords)
                        prob_pos = sigmoid(logits.detach().cpu().numpy()).reshape(-1, 1)

                        prob = np.concatenate([1 - prob_pos, prob_pos], axis=1)
                        prob_cohort.append(prob_pos)

                        # Compute probablity for all slices in the stack
                        z_unique_list = np.unique(coords.numpy()[0, :, 0])
                        prob_list = []
                        ig_list = []
                        features_agg_list = []
                        print(len(z_unique_list))
                        for z_level in z_unique_list:
                            indices = np.flatnonzero(coords[0, :, 0] == z_level)
                            data_slice = data[:, indices]
                            data_slice = data_slice.to(device)
                            coords_slice = coords[:, indices]

                            # compute attn
                            logits, _ = attn_model(data_slice, coords_slice)

                            prob_pos = sigmoid(logits.detach().cpu().numpy()).reshape(-1, 1)
                            prob = np.concatenate([1 - prob_pos, prob_pos], axis=1)

                            prob_list.append(prob)

                            # Compute agg_features
                            features_enc = encode_features(attn_model, data_slice)
                            features_agg = attend_features(attn_model, features_enc, coords_slice)
                            features_agg_list.append(features_agg.detach().cpu().numpy())

                            # Integrated gardients
                            ig = IntegratedGradients(interpret_patient)
                            data_slice.requires_grad_()

                            for target in range(1):
                                ig_attr = ig.attribute((data_slice), n_steps=50, target=target)
                                ig_attr = ig_attr.squeeze().sum(dim=1).cpu().detach()

                            ig_attr = ig_attr.view(-1, 1).numpy()
                            # ig_normalized = normalize_ig_scores(ig_attr).cpu().numpy()

                            ig_list.append(ig_attr)

                        prob_list = np.concatenate(prob_list)
                        # prob_list = softmax(prob_list, axis=1)

                        coords_list.append(coords)
                        ig_list_all.append(ig_list)
                        features_agg_list = np.concatenate(features_agg_list)

                        features_agg_all.append(features_agg_list)
                        prob_list_all.append(prob_list)

        results = {'subjects': subject_list,
                   'features_agg': features_agg_all,
                   'prob_indiv': prob_list_all,
                   'prob_cohort': prob_cohort,
                   'ig': ig_list_all,
                   'coords': coords_list}

        seed = os.path.basename(model_path).split('__')[0].split('--')[-1]
        patch_train = conf['patch']
        patch_test = conf['feats_path'].split('/')[-3]

        path_dict = {'seed': seed,
                     'train': patch_train,
                     'test': patch_test,
                     'encoder': conf['encoder'],
                     'decoder': conf['decoder'],
                     'numOfaug': conf['numOfaug'],
                     'dec_enc_dim': config_exp['decoder_enc_dim'],
                     'attn_latent_dim': config_exp['attn_latent_dim']}

        basename = 'seed--{seed}__train--{train}__test--{test}__enc--{encoder}'\
                    '__dec--{decoder}--{dec_enc_dim}--{attn_latent_dim}__aug--{numOfaug}'.format(**path_dict)

        result_path_new = os.path.join(conf['save_dir'], basename)
        os.makedirs(result_path_new, exist_ok=True)

        # Save config file
        with open(os.path.join(result_path_new, 'conf.yaml'), 'w') as f:
            yaml.dump(path_dict, f)

        with open(os.path.join(result_path_new, 'result.pkl'), 'wb') as f:
            pickle.dump(results, f)

