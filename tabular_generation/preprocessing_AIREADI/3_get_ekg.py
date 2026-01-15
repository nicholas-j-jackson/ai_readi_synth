import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PATH", type=str, help='location of the original AI-READI dataset')
args = parser.parse_args()

path = args.DATA_PATH #'../../synth_data/dataset/'
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
df = pd.read_csv(path + 'cardiac_ecg/manifest.tsv', delimiter='\t')
#df.head()

cols = ['manufacturer', 'device_model', 'modality', 'header_version', 'dataset_information', 'dataset_usage_and_license', 'machine_text', 'machine_detail_description', 'interpretation_criteriaversion', 'patient_criteriaversion', 'internalmeasurements_version', 'Time_axis', 'Amplitude', 'device_documentation_type_and_version', 'participant_id', 'participant_position', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T', 'high_pass_filter_setting', 'low_pass_filter_setting', 'notch_filter_setting', 'notch_harmonic_setting', 'artifact_filter_flag', 'hysteresis_filter_flag', 'notch_filtered', 'ac_setting', 'report_description', 'interpretation_comment_1', 'interpretation_comment_2', 'validation_id', 'validation_date']

for i in range(1,9):
    cols.extend(['comment_{}_key'.format(i), 'comment_{}_val'.format(i)])

data = {col: [] for col in cols}

for i in range(len(df)):
    _, header = wfdb.rdsamp(path+df.loc[i]['wfdb_hea_filepath'][:-4])

    for x in header['comments']:
        temp = x.split(':')
        if len(temp) == 2:
            data[temp[0]].append(temp[1].strip())
        else:
            data[temp[0]].append(np.nan)

    for j in range(1,9):
        if len(data['comment_{}_key'.format(j)]) != i+1:
            data['comment_{}_key'.format(j)].append(np.nan)

        if len(data['comment_{}_val'.format(j)]) != i+1:
            data['comment_{}_val'.format(j)].append(np.nan)

# Convert to dataframe and drop all uninformative columns
data = pd.DataFrame(data)
data = data[[col for col in data.columns if len(data[col].value_counts()) != 1 or 'comment' in col]]

# Add binary columns for normal/abnormal
data = pd.concat([data, pd.get_dummies(data['interpretation_comment_2'].apply(lambda x: x[2:-6])).astype(int)], axis=1).drop(columns=['interpretation_comment_2'])

# Make column numeric
data['participant_position'] = data['participant_position'].apply(lambda x: int(x.split(' ')[0]))

data = data[['participant_id',
       'participant_position', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS',
       'T', 'ABNORMAL', 'BORDERLINE', 'NORMAL',
       'OTHERWISE NORMAL']]
data = data.rename(columns={"participant_id": "patient_id"})
os.makedirs("ekgs", exist_ok=True)
data.to_csv('ekgs/ekg_cleaned.csv')