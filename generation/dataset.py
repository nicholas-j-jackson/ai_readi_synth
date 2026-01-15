import json
import os
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose, ToTensor


import pandas as pd
import numpy as np
class AI_READI_Dataset(Dataset):

    def __init__(self, root: str, transforms: Optional[Compose] = None, mode: Optional[str] = 'train', task: Optional[str] = 'fundus', pre_embed: Optional[bool] = True) -> None:
        self.root = root
        self.task = task
        self.pre_embed = pre_embed

        if task == 'fundus':
            self.df = pd.read_csv(root + 'resized_retinal_photography/updated_manifest.csv', index_col=0) # TODO: change to resized_retinal_oct

            self.cols = [
            'Aurora', 'Cirrus', 'Eidon', 'Maestro2', 'Spectralis', 'Triton', 
            'L', 'R', 
            'Macula', 'Macula or Optic Disc', 'Mosaic', 'Nasal', 'Optic Disc','Temporal Periphery', 'Wide Field', 'Autofluorescence',
            'Color Photography', 'Infrared Reflectance',  'AMD', 'DR', 'GL'
            ]

        elif task == 'oct':
            self.df = pd.read_csv(root + 'resized_retinal_oct/updated_manifest.csv', index_col=0) # TODO: change to resized_retinal_oct
            self.cols =  ['Cirrus', 'Maestro2', 'Spectralis', 'Triton', 'Macula', 'Optic Disc', 'Wide Field', 'L', 'R', 'AMD', 'DR', 'GL']

        else: 
            raise NotImplementedError("Invalid task: {}".format(task))
        
        print(root)
        ids = pd.read_csv(root + 'participants.tsv', delimiter='\t')
        ids = ids[ids['recommended_split'] == mode]['person_id'].values
        self.df = self.df[self.df['person_id'].isin(ids)]
        self.df['filepath'] = self.df['filepath'].apply(lambda x: x.replace('.dcm', '.jpg'))

        if task == 'oct': 
            self.df['middle_slice'] = (self.df['number_of_frames'] - 1) // 2
            self.df['filepath'] = self.df.apply(lambda x: x['filepath'].replace('.jpg', '_{}.jpg'.format(x['middle_slice'])), axis=1)

        # Works for fundus only
        self.df['anatomic_region'] = self.df['anatomic_region'].apply(lambda x: x.replace(', 6 x 6', '').replace(', 12 x 12', ''))

        # Read in clinical data
        obs = pd.read_csv(root + '/clinical_data/observation.csv', index_col=None)
        person = pd.read_csv(root + '/clinical_data/person.csv', index_col=None)

        # Identify people who have these opthalmologic conditions
        codes = {'AMD': 374028, 'DR':4174977, 'RO': 440392, 'GL': 437541, 'CAT': 4317977}
        #conditions = ['AMD', 'DR', 'RO', "GL", 'CAT']
        person = person[['person_id']]

        for condition, code in codes.items():
            person = pd.merge(person, obs[obs['qualifier_concept_id'] == code][['person_id', 'value_as_number']].rename(columns={'value_as_number': condition}), on='person_id', how='right')#.values
            
        # 23 people answered 'prefer not to answer' (coded as 777) to one of the questions about diseases
        #person = person[np.logical_not((person[conditions] == 777).any(axis=1))]

        # Merge the above clinical data with basic demographic information
        part = pd.read_csv(root + '/participants.tsv', delimiter='\t')
        #part = part[['participant_id', 'clinical_site', 'age', 'study_group', 'study_visit_date', 'recommended_split']].rename(columns={'participant_id':'person_id'})
        person = pd.merge(person, part[['person_id', 'clinical_site', 'age', 'study_group', 'study_visit_date', 'recommended_split']], on='person_id', how='left')

        # Merge clinical + demographic with image-level data
        #self.df = pd.merge(self.df, person.rename(columns={'person_id': 'participant_id'}), how='inner', on='participant_id')
        self.df = pd.merge(self.df, person, how='inner', on='person_id')

        # Create one-hot encoded columns for T2D status, imaging equipment, laterality, anatomic region, and imaging type
        self.df = pd.concat([self.df] + [pd.get_dummies(self.df[x]).astype(int) for x in ['study_group', 'manufacturers_model_name', 'laterality', 'anatomic_region', 'imaging']], axis=1)

        if self.pre_embed:
            if task == 'oct':
                self.df['filepath'] = self.df.apply(lambda x: x['filepath'].replace('.jpg', '.npy').replace('resized_retinal_oct', 'latent_oct'), axis=1)
            else:
                self.df['filepath'] = self.df.apply(lambda x: x['filepath'].replace('.jpg', '.npy').replace('resized_retinal_photography', 'latent_photography'), axis=1)


        if transforms is None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        """Return length of the dataset."""
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset element."""
        meta = self.df.iloc[idx]

        if not self.pre_embed:
            img = Image.open(self.root + meta['filepath']).convert('RGB')
        else:
            img = np.load(self.root + meta['filepath'])
            img = np.transpose(img, (1, 2, 0))
            #print(img.shape)

        img = self.transforms(img)

        dev = torch.from_numpy(np.array(meta[self.cols].values).astype(np.float32)).squeeze()
        
        
        # Human-readable label        
        human_label = meta[['manufacturers_model_name']].values.astype(str).tolist()[0] + '_' + meta[['anatomic_region']].values.astype(str).tolist()[0] + "_" + meta[['laterality']].values.astype(str).tolist()[0] + '_' + meta[['imaging']].values.astype(str).tolist()[0]

        return {'img': img, 'class_label': dev, "human_label": human_label}


