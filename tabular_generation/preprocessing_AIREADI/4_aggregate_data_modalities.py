
import pandas as pd 
import numpy as np
import warnings
from itertools import cycle
import re
import argparse 
from pathlib import Path
import os 

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 30)   
pd.set_option("display.max_rows",  50) 
pd.set_option("display.width",  None)  

def extract_condition_name(condition_str):
    if ',' in condition_str:
        return condition_str.split(',')[1].strip()
    else:
        match = re.search(r'mhoccur_(\w+)|mh_(\w+)', condition_str)
        if match:
            return match.group(1) or match.group(2)
        return condition_str

def get_conditions(DATA_PATH, PATIENT_IDS): 
    csv_path = Path(f"{DATA_PATH}/clinical_data/condition_occurrence.csv") 
    df = pd.read_csv(csv_path)
    df = df[df['person_id'].isin(PATIENT_IDS)]
    col_to_keep = ["condition_occurrence_id", "person_id", "condition_concept_id", "condition_type_concept_id", "condition_status_concept_id", "condition_source_value"]
    df = df[col_to_keep].sort_values(by=["person_id", "condition_concept_id"]).reset_index(drop=True)
    conditions = df["condition_source_value"].unique()
    df['condition'] = df['condition_source_value'].apply(extract_condition_name)
    df['condition'] = df['condition'].replace({ 'Age-related macular degeneration (AM': "Age-related macular degeneration", 'Arthritis (joint pain)': 'Arthritis', 'Cancer (any type)': 'Cancer',
       'Cataracts (in one or both eyes)': 'Cataracts (1+ eyes)',
       'Chronic pulmonary (lung) problems (E': 'Chronic pullmonary problems',
       'Circulation problems (Examples: art': 'Circulation problems',
       "Dementia (Examples: Alzheimer's Disea": 'Dementia/Alzheimers',
       'Diabetic retinopathy (in one or both': 'Diabetic retinopathy (1+)',
       'Digestive problems (Examples: stomach': 'Digestive problems', 
       'Do you use marijuana now?': 'Marijuana user',
       'Dry eye (in one or both eyes)': 'Dry eye (1+)',
       'Elevated A1C levels (elevated blood sugar': 'Elevated A1C',
       'Glaucoma (in one or both eyes)': 'Glaucoma (1+)', 
    'Mild cognitive impairment (known as': 'Mild cognitive impairmen',
       'Other heart issues (Examples: pace': 'Other heart issues (pacemaker)',
       'Retinal vascular occlusion ("stroke': 'Retinal vascular occlusion', 
       'Urinary problems (Examples: urinary t': 'Urinary problems'})
    condition_matrix = df.pivot_table(index='person_id', columns='condition', values='condition_occurrence_id', aggfunc='count', fill_value=0)
    # Convert to binary
    condition_matrix = (condition_matrix > 0).astype(int)
    condition_matrix = condition_matrix.reset_index().rename(columns={'person_id': 'patient_id'})
    condition_matrix = condition_matrix.fillna(0).astype(int)
    os.makedirs('conditions', exist_ok=True)
    csv_path = Path("conditions/conditions.csv")
    condition_matrix.to_csv(csv_path)
    txt_path = csv_path.with_suffix(".txt")
    txt_path.write_text(condition_matrix.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))
    return condition_matrix 

def get_measurements(DATA_PATH, PATIENT_IDS): 
    csv_path = Path(f"{DATA_PATH}/clinical_data/measurement.csv") 
    df = pd.read_csv(csv_path)
    df = df[df['person_id'].isin(PATIENT_IDS)]
    col_to_keep = ["person_id", "measurement_date", "measurement_source_value", "unit_source_value", "value_as_number", "value_source_value"]
    df = df[col_to_keep].sort_values(by=["person_id", "measurement_source_value"]).reset_index(drop=True)
    df = df.rename(columns={'person_id': 'patient_id'})
    #
    #df['value_source_value'] = pd.to_numeric(df['value_source_value'], errors='coerce')
    df['value_as_number'] = pd.to_numeric(df['value_as_number'], errors='coerce')
    df['value_source_value'] = pd.to_numeric(df['value_source_value'], errors='coerce')
    df['value'] = df['value_as_number'].values
    df.loc[df['value'].isnull(), 'value'] = df.loc[df['value'].isnull(), 'value_source_value']

    #df['measurement_column'] = df['measurement_source_value'] + '_value'
    df.loc[df['unit_source_value'].notna(), 'measurement_column'] = (df.loc[df['unit_source_value'].notna(), 'measurement_source_value'] + '_value (' + df.loc[df['unit_source_value'].notna(), 'unit_source_value'] + ')')
    df.loc[df['measurement_column'].isnull(), 'measurement_column'] = df.loc[df['measurement_column'].isnull(), 'measurement_source_value']
    pivoted_df = df.pivot_table(
        index=['patient_id', 'measurement_date'], 
        columns='measurement_column', 
        values='value',
        aggfunc='first'  #
    ).reset_index()
    pivoted_df.columns.name = None
    print("Available measurements:")
    print(pivoted_df.columns)
    print([col for col in pivoted_df.columns if col not in ['patient_id', 'measurement_date']])
    print(f"\nShape: {pivoted_df.shape}")
    print(pivoted_df.head())
    os.makedirs('measurements', exist_ok=True)
    csv_path = Path("measurements/measurements.csv")
    pivoted_df.to_csv(csv_path, index=False)
    txt_path = csv_path.with_suffix(".txt")
    txt_path.write_text(pivoted_df.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))
    return pivoted_df



def pick_best(g, prefer='normal', prefer_position=(0,30,60)):
    num_cols   = ['Rate','PR','QRSD','QT','QTc']
    angle_cols = ['P','QRS','T']
    flag_cols  = ['ABNORMAL','BORDERLINE','NORMAL','OTHERWISE NORMAL']
    g = g.copy()
    #drop obviously broken ECGs
    g = g[~((g['QTc'] == 0) | (g['QTc'].isna() & g['QT'].isna()))]
    #prefer normal 
    sev_weights = {'ABNORMAL':3,'BORDERLINE':2,'NORMAL':1,'OTHERWISE NORMAL':0}
    g['sev_score'] = g[flag_cols].mul([sev_weights[c] for c in flag_cols]).max(axis=1)
    if prefer == 'normal':  # flip so lower is better
        g['sev_score'] = -g['sev_score']
    #prefer 0 over 30 over 60 
    pos_order = {p:i for i,p in enumerate(prefer_position)}
    g['pos_score'] = g['participant_position'].map(pos_order).fillna(len(pos_order))
    #completeness
    g['nonnull'] = g[num_cols + angle_cols].notna().sum(axis=1)
    #final tie breaker QtC
    g = g.sort_values(['pos_score','sev_score','nonnull','QTc'], ascending=[True, True, False, False])
    return g.iloc[0]

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_PATH", required=True)
    args = parser.parse_args()

    PATIENT_IDS = pd.read_csv(f'{args.DATA_PATH}/participants.tsv', delimiter='\t')
    PATIENT_IDS = PATIENT_IDS[PATIENT_IDS[['cardiac_ecg', 'clinical_data', 'wearable_activity_monitor', 'wearable_blood_glucose']].all(axis=1)]['person_id'].values
    PATIENT_IDS = [x for x in PATIENT_IDS if x not in [1059, 1078, 1082, 1107, 1108, 1127, 1130, 1142, 1150, 1162, 1265, 1279, 1319, 1342, 1343, 1369, 1371, 1375, 1395, 1435, 1459, 1460, 1464, 1488, 1517, 1522, 1560, 1561, 1570, 1581, 1601, 1633, 1638, 1671, 1672, 1692, 1697, 1700, 1711, 1724, 1755, 1761, 1781, 1783, 1796, 4025, 4029, 4038, 4050, 4057, 4068, 4069, 4070, 4071, 4075, 4079, 4080, 4081, 4083, 4084, 4085, 4086, 4090, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4102, 4108, 4110, 4129, 4137, 4173, 4174, 4176, 4194, 4195, 4197, 4198, 4199, 4209, 4212, 4213, 4214, 4217, 4218, 4223, 4233, 4238, 4242, 4244, 4258, 4259, 4260, 4262, 4272, 4277, 4280, 4288, 4293, 4295, 4300, 4303, 4307, 4324, 4325, 4326, 4339, 4342, 4346, 4353, 4354, 4355, 4359, 4363, 4367, 4368, 4369, 4375, 4382, 4386, 4390, 4393, 4397, 4398, 4401, 4404, 4408, 4414, 4415, 4443, 4448, 4458, 4459, 4465, 4469, 4470, 4474, 4475, 4480, 4484, 4491, 4493, 4495, 4500, 4502, 4507, 4511, 4524, 4552, 4553, 4570, 4572, 4573, 4580, 4582, 4584, 4598, 4601, 4602, 4607, 4610, 4611, 4613, 4637, 4641, 4664, 4675, 4679, 4685, 7060, 7135, 7289, 7321, 7324, 7331, 7342, 7353, 7370, 7380, 7402, 7410, 7426, 7441, 7444, 7463, 7468, 7494, 7509, 7526, 7535, 7550, 7554, 7560, 7565, 7586, 7594, 7621, 7686, 7702]]

    conditions = get_conditions(args.DATA_PATH, PATIENT_IDS) 
    measurements = get_measurements(args.DATA_PATH, PATIENT_IDS) 
    
    #load wearable (contains blood glucose too), conditions, measurements and ekg 
    #conditions = pd.read_csv("conditions/conditions.csv")
    #measurements = pd.read_csv("measurements/measurements.csv")
    scalar_wearable = pd.read_csv("scalar_data/all_patients/scalar_from_aligned/all_patients_8_24.0h_3.0h_False_True.csv")
    ekg = pd.read_csv("ekgs/ekg_cleaned.csv")

    #match patients 
    scalar_wearable = scalar_wearable[scalar_wearable['patient_id'].isin(PATIENT_IDS)]
    patients = list(scalar_wearable['patient_id'])
    conditions = conditions[conditions['patient_id'].isin(patients)]
    print("COND_COLS: ",  conditions.columns.values.tolist())
    measurements = measurements[measurements['patient_id'].isin(patients)]
    print("MEAS_COLS: ",  measurements.columns.values.tolist())
    ekg = ekg[ekg['patient_id'].isin(patients)]
    print("EKG_COLS: ",  ekg.columns.values.tolist())

    #collapse duplicate measurement rows 
    measurements = measurements.drop(columns=['measurement_date'])
    #measurements = measurements.dropna()
    key = ['patient_id']
    df = measurements
    key_cols = [key] if isinstance(key, str) else list(key)
    dupes = df[df.duplicated(key_cols, keep=False)].copy()
    value_cols = [c for c in df.columns if c not in key_cols]
    dupes  = df[df.duplicated(key_cols, keep=False)].copy()
    unique = df[~df.duplicated(key_cols, keep=False)].copy()
    # Collapse each duplicate group by coalescing non-nulls column-wise
    collapsed = (dupes.groupby(key_cols)[value_cols].apply(lambda g: g.bfill().ffill().iloc[0]).reset_index())
    df_merged = pd.concat([unique, collapsed], ignore_index=True)
    assert not df_merged.duplicated(key_cols, keep=False).any()
    measurements = df_merged 

    #remove duplicates for ekg: prefer the Normal score, drop any rows with Nan measurements, prefer position 0 
    ekg = (ekg.groupby('patient_id', group_keys=False).apply(pick_best).reset_index(drop=True))
    ekg = ekg.drop(columns=['sev_score', 'pos_score', 'nonnull'])

    #merge wearables with conditions, measurements and ekgs 
    combined_all = scalar_wearable.merge(conditions, on='patient_id', how='left')
    combined_all = combined_all.merge(measurements, on='patient_id', how='left')
    combined_all = combined_all.merge(ekg, on='patient_id', how='left')
    combined_all = combined_all.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"])#.dropna()
    
    drop_cols = ['import_ldl_cholesterol, Cholesterol in LDL [Mass/', 'import_glucose, Glucose [Mass/volume] in Serum or', 'import_crp_hs, C reactive protein [Mass/volume] i', 'import_hba1c, Hemoglobin A1c/Hemoglobin.total in ', 'import_creatinine, Creatinine [Mass/volume] in Se', 'import_carbon_dioxide_total, Carbon dioxide, tota', 'import_bun, Urea nitrogen [Mass/volume] in Serum ', 'import_alt_got, Alanine aminotransferase [Enzymat', 'import_ast_got, Aspartate aminotransferase [Enzym', 'import_chloride, Chloride [Moles/volume] in Serum', 'import_calcium, Calcium [Mass/volume] in Serum or', 'import_albumin, Albumin [Mass/volume] in Serum or', 'import_alkaline_phosphatase, Alkaline phosphatase', 'import_potassium, Potassium [Moles/volume] in Ser', 'import_nt_probnp, Natriuretic peptide.B prohormon', 'import_hdl_cholesterol, Cholesterol in HDL [Mass/', 'import_troponin_t, Troponin T.cardiac [Mass/volum', 'import_total_cholesterol, Cholesterol [Mass/volum', 'import_protein_total, Protein [Mass/volume] in Se', 'import_triglycerides, Triglyceride [Mass/volume] ', 'import_urine_albumin, Albumin [Mass/volume] in Ur', 'import_sodium, Sodium [Moles/volume] in Serum or ', 'import_urine_creatinine, Creatinine [Mass/volume]']
    combined_all = combined_all.drop(columns=drop_cols).dropna()

    combined_all.to_csv("ai_readi_full_data.csv")
    print(combined_all.shape)
    print(list(combined_all.columns))
    
if __name__ == "__main__":
    main()