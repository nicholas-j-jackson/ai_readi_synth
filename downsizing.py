#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import pydicom as dcm  
import tqdm as tqdm
import os
from PIL import Image


folder = '/mnt/disks/data/87378b13-a9de-45cd-853f-239dbe0cdd64/dataset/'


# In[2]:


## RETINAL FUNDUS IMAGES


# In[3]:


df = pd.read_csv(folder + 'retinal_photography/manifest.tsv', delimiter='\t')


def resize_fundus(filepath):
    try:
        img = dcm.dcmread(folder + filepath).pixel_array
        resized = Image.fromarray(img).resize((512, 512))

        if not os.path.exists(folder + 'resized_retinal_photography/' + "/".join(filepath.split('/')[2:-1])):
            os.makedirs(folder + 'resized_retinal_photography/' + "/".join(filepath.split('/')[2:-1]))

        resized.save(folder + 'resized_retinal_photography/' + "/".join(filepath.split('/')[2:]).replace('.dcm', '.jpg'))
        return 1
    except: 
        return 0


from joblib import Parallel, delayed
#results = Parallel(n_jobs=8)(delayed(resize_fundus)(temp.filepath) for ind, temp in tqdm.tqdm(df.iterrows(), total=len(df), smoothing=0))



df.rename(columns={'height':'original_height', 'width':'original_width'})
df['height'] = 512
df['width'] = 512

df['filepath'] = df['filepath'].apply(lambda x: x.replace('retinal_photography', 'resized_retinal_photography').replace('.dcm', '.jpg'))

#df.to_csv(folder + '/resized_retinal_photography/updated_manifest.csv')


# In[ ]:





# In[ ]:





# In[4]:


## Retinal OCT
# Following the process in https://www.nature.com/articles/s41598-023-41362-4
# we select the middle slice and resize 


# In[5]:


df = pd.read_csv(folder + 'retinal_oct/manifest.tsv', delimiter='\t')


df = pd.read_csv(folder + 'retinal_oct/manifest.tsv', delimiter='\t')


def resize_oct(filepath):
    try:
    
        imgs = dcm.dcmread(folder + filepath).pixel_array
        
        if not os.path.exists(folder + 'resized_retinal_oct/' + "/".join(filepath.split('/')[2:-1])):
            os.makedirs(folder + 'resized_retinal_oct/' + "/".join(filepath.split('/')[2:-1]))
        
        for slice in range(len(imgs)):
            resized = Image.fromarray(imgs[slice]).resize((512, 512))
            resized.save(folder + 'resized_retinal_oct/' + "/".join(filepath.split('/')[2:]).replace('.dcm', '_' + str(slice) + '.jpg'))
            #print((folder + 'resized_retinal_oct/' + "/".join(filepath.split('/')[2:])).replace('.dcm', '_' + str(slice) + '.jpg'))
        return 1
    except Exception as ex:
        print(ex)
        return 0


from joblib import Parallel, delayed
results = Parallel(n_jobs=8)(delayed(resize_oct)(temp.filepath) for ind, temp in tqdm.tqdm(df.iterrows(), total=len(df), smoothing=0))


df.rename(columns={'height':'original_height', 'width':'original_width'})
df['height'] = 512
df['width'] = 512


df['filepath'] = df['filepath'].apply(lambda x: x.replace('retinal_oct', 'resized_retinal_oct'))

df.to_csv(folder + '/resized_retinal_oct/updated_manifest.csv')
