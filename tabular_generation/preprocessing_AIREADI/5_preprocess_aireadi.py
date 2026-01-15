import pandas as pd 
import os 
import numpy as np
import argparse 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

seed = 0 #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#all conditions except elevated A1C and prediabetes as that is too highly correlated with diabetes AND ekg categorical diagnoses 
CAT_COLS = ['Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Dementia/Alzheimers', 'Diabetic retinopathy (1+)', 
            'Digestive problems', 'Dry eye (1+)', 'Elevated A1C', 'Glaucoma (1+)', 'Hearing impairment', 'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 
            'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis', 'Other heart issues (pacemaker)', 'Other neurological conditions', "Parkinson's disease", 'Pre-diabetes', 'Retinal vascular occlusion', 
            'Stroke', 'Type 2 Diabetes', 'Urinary problems', 'lettera', 'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL']


# Used to fix weridness in naming. These columns are not actually used (many duplicates)
MEAS_COLS = ['bmi_vsorres, BMI_value ( )', 'bp1_diabp_vsorres, Diastolic (mmHg)_value ( )', 'bp1_sysbp_vsorres, Systolic (mmHg)_value ( )', 'bp2_diabp_vsorres, Diastolic (mmHg)_value ( )', 'bp2_sysbp_vsorres, Systolic (mmHg)_value ( )', 'clock_visuospatial_executive', 'clock_visuospatial_executive_time', 'cube_visuospatial_executive', 'cube_visuospatial_executive_time', 'delayed_recall_with_no_clue', 'delayed_recall_with_no_clue_time', 'digitspan', 'digitspan_time', 'height_vsorres, Height (cm)_value ( )', 'hip_vsorres, Hip Circumference (cm)_value ( )', 'import_a_g_ratio, Albumin/Globulin ratio', 'import_albumin, Albumin [Mass/volume] in Serum or', 'import_albumin, Albumin [Mass/volume] in Serum or_value (g/dL)', 'import_alkaline_phosphatase, Alkaline phosphatase', 'import_alkaline_phosphatase, Alkaline phosphatase_value (IU/L)', 'import_alt_got, Alanine aminotransferase [Enzymat', 'import_alt_got, Alanine aminotransferase [Enzymat_value (IU/L)', 'import_ast_got, Aspartate aminotransferase [Enzym', 'import_ast_got, Aspartate aminotransferase [Enzym_value (IU/L)', 'import_bilirubin_total, Bilirubin.total [Mass/vol', 'import_bun, Urea nitrogen [Mass/volume] in Serum ', 'import_bun, Urea nitrogen [Mass/volume] in Serum _value (mg/dL)', 'import_buncreatinineratio, BUN/Creatinine ratio', 'import_c_peptide, C peptide [Mass/volume] in Seru_value (ng/L)', 'import_calcium, Calcium [Mass/volume] in Serum or', 'import_calcium, Calcium [Mass/volume] in Serum or_value (mEq/L)', 'import_carbon_dioxide_total, Carbon dioxide, tota', 'import_carbon_dioxide_total, Carbon dioxide, tota_value (mEq/L)', 'import_chloride, Chloride [Moles/volume] in Serum', 'import_chloride, Chloride [Moles/volume] in Serum_value (mEq/L)', 'import_creatinine, Creatinine [Mass/volume] in Se', 'import_creatinine, Creatinine [Mass/volume] in Se_value (mg/dL)', 'import_crp_hs, C reactive protein [Mass/volume] i', 'import_crp_hs, C reactive protein [Mass/volume] i_value (mg/L)', 'import_globulin_total, Globulin [Mass/volume] in ', 'import_glucose, Glucose [Mass/volume] in Serum or', 'import_glucose, Glucose [Mass/volume] in Serum or_value (mg/dL)', 'import_hba1c, Hemoglobin A1c/Hemoglobin.total in ', 'import_hba1c, Hemoglobin A1c/Hemoglobin.total in _value (%)', 'import_hdl_cholesterol, Cholesterol in HDL [Mass/', 'import_hdl_cholesterol, Cholesterol in HDL [Mass/_value (mg/dL)', 'import_insulin, Insulin [Units/volume] in Serum o_value (ng/L)', 'import_ldl_cholesterol, Cholesterol in LDL [Mass/', 'import_ldl_cholesterol, Cholesterol in LDL [Mass/_value (mg/dL)', 'import_nt_probnp, Natriuretic peptide.B prohormon', 'import_nt_probnp, Natriuretic peptide.B prohormon_value (pg/mL)', 'import_potassium, Potassium [Moles/volume] in Ser', 'import_potassium, Potassium [Moles/volume] in Ser_value (mEq/L)', 'import_protein_total, Protein [Mass/volume] in Se', 'import_protein_total, Protein [Mass/volume] in Se_value (g/dL)', 'import_sodium, Sodium [Moles/volume] in Serum or ', 'import_sodium, Sodium [Moles/volume] in Serum or _value (mEq/L)', 'import_total_cholesterol, Cholesterol [Mass/volum', 'import_total_cholesterol, Cholesterol [Mass/volum_value (mg/dL)', 'import_triglycerides, Triglyceride [Mass/volume] ', 'import_triglycerides, Triglyceride [Mass/volume] _value (mg/dL)', 'import_troponin_t, Troponin T.cardiac [Mass/volum', 'import_troponin_t, Troponin T.cardiac [Mass/volum_value (ng/L)', 'import_urine_albumin, Albumin [Mass/volume] in Ur', 'import_urine_albumin, Albumin [Mass/volume] in Ur_value (_g/mL)', 'import_urine_creatinine, Creatinine [Mass/volume]', 'import_urine_creatinine, Creatinine [Mass/volume]_value (_g/mL)', 'lbscat_a1c, Hemoglobin - g/dL_value ( )', 'lbscat_hct, Hematocrit\xa0 - %_value ( )', 'lbscat_mch, MCH - pg_value ( )', 'lbscat_mchc, MCHC - g/dL_value ( )', 'lbscat_mcv, MCV - fL_value ( )', 'lbscat_plt, Platelets - x10E3/µL_value ( )', 'lbscat_rbc, Red Blood Cells (RBC) - x10E6/µL_value ( )', 'lbscat_rdw, RDW - %_value ( )', 'lbscat_wbc, White Blood Cells (WBC) - x10E3/µL_value ( )', 'lettera', 'lettera_time', 'memory_trial1', 'memory_trial1_time', 'memory_trial2', 'memory_trial2_time', 'mlcsodfcl, OD: Value of final correct letter_value ( )', 'mlcsodlog, OD: Log Contrast Sensitivity_value ( )', 'mlcsodmiss, OD: Number of misses prior to stoppin_value ( )', 'mlcsosfcl, OS: Value of final correct letter_value ( )', 'mlcsoslog, OS: Log Contrast Sensitivity_value ( )', 'mlcsosmiss, OS: Number of misses prior to stoppin_value ( )', 'moca_abstraction', 'moca_abstraction_time', 'moca_combined_mis_score', 'moca_orientation', 'moca_orientation_time', 'moca_total_score', 'msslffl, Left Foot - Felt:_value ( )', 'mssrffl, Right Foot - Felt:_value ( )', 'naming', 'naming_time', 'plcsodfcl, OD: Value of final correct letter_value ( )', 'plcsodlog, OD: Log Contrast Sensitivity_value ( )', 'plcsodmiss, OD: Number of misses prior to stoppin_value ( )', 'plcsosfcl, OS: Value of final correct letter_value ( )', 'plcsoslog, OS: Log Contrast Sensitivity_value ( )', 'plcsosmiss, OS: Number of misses prior to stoppin_value ( )', 'pulse_vsorres, Heart Rate (bpm)_value ( )', 'pulse_vsorres_2, Heart Rate (bpm)_value ( )', 'repetition', 'repetition_time', 'subtraction', 'subtraction_time', 'trails_visuospatial_executive', 'trails_visuospatial_executive_time', 'viaodaxi, OD - Autorefractor - Axis_value ( )', 'viaodcyl, OD - Autorefractor - Cylinder_value ( )', 'viaodmlog, LLVA Letter Score - Mesopic VA - OD_value ( )', 'viaodmscore, Mesopic LogMAR OD Score_value ( )', 'viaodplog, VA Letter Score - Photopic VA - OD_value ( )', 'viaodpscore, Photopic LogMAR OD Score_value ( )', 'viaodsph, OD - Autorefractor - Sphere_value ( )', 'viaosaxi, OS - Autorefractor - Axis_value ( )', 'viaoscyl, OS - Autorefractor - Cylinder_value ( )', 'viaosmlog, LLVA Letter Score - Mesopic VA - OS_value ( )', 'viaosmscore, Mesopic LogMAR OS Score_value ( )', 'viaosplog, VA Letter Score - Photopic VA - OS_value ( )', 'viaospscore, Photopic LogMAR OS Score_value ( )', 'viaossph, OS - Autorefractor - Sphere_value ( )', 'waist_vsorres, Waist Circumference (cm)_value ( )', 'weight_vsorres, Weight (kilograms)_value ( )', 'whr_vsorres, Waist to Hip Ratio (WHR)_value ( )']
names = [x.replace('_value ( )', '') for x in MEAS_COLS]
names = [x.split(', ')[1] if ', ' in x else x for x in names]
renaming = {MEAS_COLS[i]: names[i] for i in range(len(MEAS_COLS))}


#same wearables, NEW measurements and NEW ekgs 
CONT_COLS =  ['heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps',  'resting_heart_rate', 'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs','act_walking_hrs', 'act_sedentary_hrs', 
              'BMI', 'Diastolic (mmHg)', 'Systolic (mmHg)', 'clock_visuospatial_executive', 'clock_visuospatial_executive_time', 'cube_visuospatial_executive', 'cube_visuospatial_executive_time', 'delayed_recall_with_no_clue', 'delayed_recall_with_no_clue_time', 'digitspan', 'digitspan_time', 
        'Height (cm)', 'Hip Circumference (cm)', 'Albumin/Globulin ratio', 'Albumin [Mass/volume] in Serum or_value (g/dL)', 'Alkaline phosphatase_value (IU/L)', 'Alanine aminotransferase [Enzymat_value (IU/L)', 'Aspartate aminotransferase [Enzym_value (IU/L)',
        'Bilirubin.total [Mass/vol', 'Urea nitrogen [Mass/volume] in Serum _value (mg/dL)', 'BUN/Creatinine ratio', 'C peptide [Mass/volume] in Seru_value (ng/L)', 'Calcium [Mass/volume] in Serum or_value (mEq/L)', 'Carbon dioxide', 'Chloride [Moles/volume] in Serum_value (mEq/L)', 
        'Creatinine [Mass/volume] in Se_value (mg/dL)', 'C reactive protein [Mass/volume] i_value (mg/L)', 'Globulin [Mass/volume] in ', 'Glucose [Mass/volume] in Serum or_value (mg/dL)', 'Hemoglobin A1c/Hemoglobin.total in _value (%)', 'Cholesterol in HDL [Mass/_value (mg/dL)', 
        'Insulin [Units/volume] in Serum o_value (ng/L)', 'Cholesterol in LDL [Mass/_value (mg/dL)', 'Natriuretic peptide.B prohormon_value (pg/mL)', 'Potassium [Moles/volume] in Ser_value (mEq/L)', 'Protein [Mass/volume] in Se_value (g/dL)', 'Sodium [Moles/volume] in Serum or _value (mEq/L)', 
        'Cholesterol [Mass/volum_value (mg/dL)', 'Triglyceride [Mass/volume] _value (mg/dL)', 'Troponin T.cardiac [Mass/volum_value (ng/L)', 'Albumin [Mass/volume] in Ur_value (_g/mL)', 'Creatinine [Mass/volume]_value (_g/mL)', 'Hemoglobin - g/dL', 'Hematocrit\xa0 - %', 'MCH - pg', 'MCHC - g/dL', 
        'MCV - fL', 'Platelets - x10E3/µL', 'Red Blood Cells (RBC) - x10E6/µL', 'RDW - %', 'White Blood Cells (WBC) - x10E3/µL', 'lettera_time', 'memory_trial1', 'memory_trial1_time', 'memory_trial2', 'memory_trial2_time', 'moca_abstraction', 'moca_abstraction_time', 'moca_combined_mis_score', 
        'moca_orientation', 'moca_orientation_time', 'moca_total_score', 'naming', 'naming_time', 'repetition', 'repetition_time', 'subtraction', 'subtraction_time', 'trails_visuospatial_executive', 'trails_visuospatial_executive_time', 'Waist Circumference (cm)', 'Weight (kilograms)', 
        'Waist to Hip Ratio (WHR)', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T']

# columns that were measured twice (will average trials)
multi_cols = ['Diastolic (mmHg)', 'Systolic (mmHg)']


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--SAVE_PATH", required=True)
    args = parser.parse_args()
    save_folder = args.SAVE_PATH

    cat_cols = CAT_COLS
    continuous_cols = CONT_COLS
    data_path = "ai_readi_full_data.csv" 

    stats = [] 
    print("Num continuous:", len(CONT_COLS))
    print("Num categorical:", len(CAT_COLS))

    #os.makedirs(save_folder, exist_ok=True)
    data = pd.read_csv(data_path, index_col=0)
    data = data.rename(columns={'Type II Diabetes': 'Type 2 Diabetes', 'Bilirubin.total [Mass/vol': 'Bilirubin Total [Mass/vol'})
    data = data.rename(columns=renaming)

    for col in multi_cols:
        data[col+'_avg'] = data[col].mean(axis=1)
        data = data.drop(columns=col)
        data = data.rename(columns={col+'_avg':col})


    #order columns and drop na
    data = data[cat_cols + continuous_cols + ["patient_id", "day"]]
    data = data.dropna() # should be no NaNs left at this stage anyway
    
    #save unnormalized data 
    data3 = data.drop(columns=["day"])
    data3.to_csv(save_folder + '/original_data_with_patients.csv', index=False)
    
    #normalize columns 
    min_max_log = {}
    for col in continuous_cols:
        col_value = np.array(data[col])
        min_max_log[col] = [np.min(col_value), np.max(col_value)]
        norm_col_value = (col_value - min_max_log[col][0]) / (min_max_log[col][1] - min_max_log[col][0])
        data[col] = list(norm_col_value)
    
    #save min max log 
    np.save(save_folder+ '/min_max_log.npy', min_max_log)
    
    #save normalized data with patients 
    data_patient = data.drop(columns=["day"])
    data_patient = data_patient.to_csv(save_folder + '/preprocessed_data_with_patients.csv', index=False)
    
    #save normalized data without patients 
    data_patient = data.drop(columns=["patient_id"])
    data_patient.to_csv(save_folder + '/preprocessed_data_no_patients.csv', index=False)
    
    #split training testing by patient to avoid leakage 
    data_full = data.sample(frac=1, random_state=seed).reset_index(drop=True) 

    def split_by_group_and_label(df):
        split = pd.read_csv('/data/7TB/nick/ai_readi_v3/participants.tsv', delimiter='\t', index_col=None)
        patients = pd.merge(df, split[['recommended_split', 'person_id']].rename(columns={'person_id': 'patient_id'}), on='patient_id')

        train_df = patients[patients['recommended_split'] != 'test'].drop(columns=['recommended_split'])
        test_df = patients[patients['recommended_split'] == 'test'].drop(columns=['recommended_split'])
        
        return train_df, test_df

    training_data_df, testing_data_df = split_by_group_and_label(data_full)
    
    print(data_full.shape, training_data_df.shape, testing_data_df.shape)

    #save normalized training and testing 
    training_data_df = training_data_df.drop(columns=["patient_id", "day"])
    testing_data_df = testing_data_df.drop(columns=["patient_id", "day"])
    training_data_df.to_csv(save_folder + '/normalized_training_data.csv', index=False)
    testing_data_df.to_csv(save_folder + '/normalized_testing_data.csv', index=False)
    
    #also save unnormalized training and testing  
    min_max_log = np.load(save_folder + '/min_max_log.npy', allow_pickle=True).item()
    for key, min_max in min_max_log.items():
        min_, max_ = min_max[0], min_max[1]
        col_values = np.array(training_data_df[key])
        training_data_df[key] = (1 - col_values)*min_ + col_values*max_
        col_values = np.array(testing_data_df[key])
        testing_data_df[key] = (1 - col_values)*min_ + col_values*max_
    training_data_df.to_csv(save_folder + '/original_training_data.csv', index=False)
    testing_data_df.to_csv(save_folder + '/original_testing_data.csv', index=False)

if __name__ == "__main__":
    main()