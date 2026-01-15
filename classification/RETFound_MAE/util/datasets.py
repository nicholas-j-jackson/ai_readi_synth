# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from typing import Dict, Optional
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

    def __init__(self, root, args, classes: Optional[Compose] = None, split: Optional[str] = None, synth=False) -> None:
        self.root = root
        self.classes = classes      
        self.synth = synth

        if self.synth:      
            self.df = collect_metadata_synth(self.root, classes, split, args)
        else:      
            self.df = collect_metadata_real(self.root, classes, split, args)


        # Remove null or unknown labels
        self.df = self.df[np.logical_not(self.df[classes].isnull().any(axis=1))]
        self.df = self.df[np.logical_not((self.df[self.classes] == 777).any(axis=1))]


        # Training transforms
        if split is not None and split == 'train' and not args.extract:
            self.transforms = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        
        # Validation and testing transforms
        else:
            t = []
            size = int(224)
            t.append(
                Resize(size, interpolation=InterpolationMode.BICUBIC),
            )
            t.append(CenterCrop(224))
            t.append(ToTensor())
            t.append(Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            self.transforms = Compose(t)

    def __len__(self):
        """Return length of the dataset."""
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset element."""
        meta = self.df.iloc[idx]

        if not self.synth:
            img = Image.open(self.root + meta['filepath']).convert('RGB')
        else: 
            img = Image.fromarray(pydicom.dcmread(self.root + meta['filepath']).pixel_array).convert('RGB')

        img = self.transforms(img)

        if self.classes is not None:
            label = torch.from_numpy(np.array(meta[self.classes].values, dtype=np.float32))
            return {'img': img, 'label': label,  'path': meta['filepath']}
        else: 
            return {'img': img, 'path': meta['filepath']}
        
