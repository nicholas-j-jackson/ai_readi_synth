# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from typing import List
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
import pandas as pd
import numpy as np
import torch
import pydicom

def collect_metadata_real(path, classes, split, args):

    if args.modality == 'fundus':
        path += 'resized_retinal_photography/'
    else:
        path += 'resized_retinal_oct/'

    df = pd.read_csv(path+'updated_manifest.csv', index_col=0)
    df['filepath'] = df['filepath'].apply(lambda x: x.replace('.dcm', '.jpg'))

    # 
    if args.modality == 'oct': 
        df['middle_slice'] = (df['number_of_frames'] - 1) // 2
        df['filepath'] = df.apply(lambda x: x['filepath'].replace('.jpg', '_{}.jpg'.format(x['middle_slice'])), axis=1)
        #df['anatomic_region'] = df['anatomic_region'].apply(lambda x: x.replace(', 6 x 6', '').replace(', 12 x 12', '')) 


    # Read in clinical data
    obs = pd.read_csv("/".join(path.split('/')[0:-2]) + '/clinical_data/observation.csv', index_col=None)
    person = pd.read_csv("/".join(path.split('/')[0:-2]) + '/clinical_data/person.csv', index_col=None)
    person = person[['person_id']]

    # Identify people who have these opthalmologic conditions
    codes = {'AMD': 374028, 'DR':4174977, 'GL': 437541}

    for condition in classes:
        code = codes[condition]
        person = pd.merge(person, obs[obs['qualifier_concept_id'] == code][['person_id', 'value_as_number']].rename(columns={'value_as_number': condition}), on='person_id', how='right')#.values
        
    # Merge the above clinical data with basic demographic information
    part = pd.read_csv("/".join(path.split('/')[0:-2]) + '/participants.tsv', delimiter='\t')
    #part = part.rename(columns={'participant_id':'person_id'})
    person = pd.merge(person, part[['person_id', 'clinical_site', 'age', 'study_group', 'study_visit_date', 'recommended_split']], on='person_id', how='left')
    
    # Merge clinical + demographic with image-level data
    #df = pd.merge(df, person.rename(columns={'person_id': 'participant_id'}), how='inner', on='participant_id')
    df = pd.merge(df, person, how='inner', on='person_id')

    # Create one-hot encoded columns for T2D status and imaging equipment
    #df = pd.concat([df, pd.get_dummies(df['study_group']).astype(int), pd.get_dummies(df['manufacturers_model_name']).astype(int)], axis=1)

    # 
    if split is not None: 
        #if args.cross_validate:
        #    df = df[df['cv_{}'.format(args.seed)] == split]
        #else:
        df = df[df['recommended_split'] == split]

    return df



def collect_metadata_synth(path, classes, split, args):

    if args.modality == 'fundus':
        path += 'retinal_photography/'
    else:
        path += 'retinal_oct/'

    df = pd.read_csv(path+'manifest.tsv', delimiter='\t', index_col=0)

    # Read in clinical data
    obs = pd.read_csv("/".join(path.split('/')[0:-2]) + '/clinical_data/observation.csv', index_col=None)
    person = obs[['person_id']].iloc[0:len(obs)//3]

    # Identify people who have these opthalmologic conditions
    codes = {'AMD': 374028, 'DR':4174977, 'GL': 437541}

    for condition in classes:
        code = codes[condition]
        person = pd.merge(person, obs[obs['qualifier_concept_id'] == code][['person_id', 'value_as_number']].rename(columns={'value_as_number': condition}), on='person_id', how='right')#.values
        
    # Create split
    person['recommended_split'] = pd.cut(range(len(person)), bins=[0, 0.8*len(person), 0.9*len(person), len(person)], labels=['train', 'val', 'test'], right=False)

    # Merge clinical + demographic with image-level data
    df = pd.merge(df, person.rename(columns={'person_id': 'participant_id'}), how='inner', on='participant_id')

    
    # 
    if split is not None: 
        df = df[df['recommended_split'] == split]

    return df


class RetinalImageDataset(Dataset):

    def __init__(self, root: str, args, classes: List[str], mode: str, synth: bool = False) -> None:
        self.root = root
        self.args = args
        self.classes = classes
        self.mode = mode
        self.synth = synth

        if synth:
            self.df = collect_metadata_synth(root, classes, mode, args)
        else:
            self.df = collect_metadata_real(root, classes, mode, args)

        self.transforms = Compose([
            Resize((args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ])
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        meta = self.df.iloc[idx]
        filepath = self.root + meta['filepath']

        if self.synth:
            dicom = pydicom.dcmread(filepath)
            img = Image.fromarray(dicom.pixel_array).convert('RGB')
        else:
            img = Image.open(filepath).convert('RGB')
        img = self.transforms(img)

        label = torch.from_numpy(
            np.array(meta[self.classes].values).astype(np.float32)).squeeze()

        return {'img': img, 'label': label}