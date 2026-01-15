from dataclasses import dataclass, asdict, replace, field
import json, pathlib
from typing import Optional, List

@dataclass(frozen=True)
class HyperParams_Search:
    #---------------USER INPUT---------------#
    #-----SAVE LOGISTICS-----#
    #where the results get saved
    OUT_DIR: Optional[str] = None   
    #where the data folder is located 
    DATA_PATH:  Optional[str] = None   
    #where the summary of results gets saved
    RESULT_CSV:  Optional[str] = None   
    #tag for above summary file 
    RUN_NAME: Optional[str] = None   
    #options: AI-READI-OG, AI-READI-FULL, MIMIC
    DATASET:  Optional[str] = None   

    #-----TRAINING-----# 
    SEED: int = 0 
    ITERS: int = 20000
    DATA_SIZE: float = 0.9
    TRAIN: bool = False

    #------------AUTOMATICALLY CONFIGURED------------#
    #-----DATA LOGISTICS-----#
    #where min max file gets saved
    NPY_PATH:  Optional[str] = None   
    #list of numeric columns (IN ORDER)
    NUM_COLS: List[str] = field(default_factory=list)
    #list of categorical columns (IN ORDER)
    CAT_COLS: List[str] = field(default_factory=list)
    #length of cat_cols
    CAT_DIM: int = 0
    #target label for classification evaluation 
    LABEL: str = "Type 2 Diabetes"

    #-----MODEL PARAMETERS-----#
    DEVICE: str = 'cpu'
    BATCH: int = 256 
    NOISE_DIM: int = 32  
    GRADIENT_PENALTY: int = 5
    G_LR: float = 0.0001 
    G_H: int = 64 
    D_LR: float = 1e-05
    D_H: int = 128 
    DISC_STEPS: int = 5 
    NUM_SAMPLES: int = 5000
    
    #PPO SPECIFIC  
    USE_TANH: bool = True  
    #number of times to iterate over PPO training loop (gen steps)
    PPO_EPOCHS: int = 3 
    MEAN_PENALTY_SCALE: float = 0.2 
    VF_COEF: float = 0.5 #default in PPO, pretty standard
    CLIP_EPS: float = 0.1 #can also try 0.2 
    ENT_BETA: float = 1e-3 #can also try 0.01 
    #simple clip on tanh to avoid bugs 
    EPS: float = 1e-6 

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=2))
    def override(self, **kwargs):
        return replace(self, **kwargs)


# Hard-coded the best hyperparams for ease of use
@dataclass(frozen=True)
class HyperParams_Best:
    #---------------USER INPUT---------------#
    #-----SAVE LOGISTICS-----#
    #where the results get saved
    OUT_DIR: Optional[str] = None   
    #where the data folder is located 
    DATA_PATH:  Optional[str] = None   
    #where the summary of results gets saved
    RESULT_CSV:  Optional[str] = None   
    #tag for above summary file 
    RUN_NAME: Optional[str] = None   
    #options: AI-READI-OG, AI-READI-FULL, MIMIC
    DATASET:  Optional[str] = None   

    #-----TRAINING-----# 
    SEED: int = 0 
    ITERS: int = 30000
    DATA_SIZE: float = 0.9
    TRAIN: bool = False


    #------------AUTOMATICALLY CONFIGURED------------#
    #-----DATA LOGISTICS-----#
    #where min max file gets saved
    NPY_PATH:  Optional[str] = None   
    #list of numeric columns (IN ORDER)
    CAT_COLS:  List[str] = field(default_factory=lambda: ['Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Dementia/Alzheimers', 'Diabetic retinopathy (1+)', 
            'Digestive problems', 'Dry eye (1+)', 'Elevated A1C', 'Glaucoma (1+)', 'Hearing impairment', 'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 
            'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis', 'Other heart issues (pacemaker)', 'Other neurological conditions', "Parkinson's disease", 'Pre-diabetes', 'Retinal vascular occlusion', 
            'Stroke', 'Type 2 Diabetes', 'Urinary problems', 'lettera', 'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL'])
    
    #list of categorical columns (IN ORDER)
    NUM_COLS:  List[str] = field(default_factory=lambda: ['heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps',  'resting_heart_rate', 'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs','act_walking_hrs', 'act_sedentary_hrs', 
              'BMI', 'Diastolic (mmHg)', 'Systolic (mmHg)', 'clock_visuospatial_executive', 'clock_visuospatial_executive_time', 'cube_visuospatial_executive', 'cube_visuospatial_executive_time', 'delayed_recall_with_no_clue', 'delayed_recall_with_no_clue_time', 'digitspan', 'digitspan_time', 
        'Height (cm)', 'Hip Circumference (cm)', 'Albumin/Globulin ratio', 'Albumin [Mass/volume] in Serum or_value (g/dL)', 'Alkaline phosphatase_value (IU/L)', 'Alanine aminotransferase [Enzymat_value (IU/L)', 'Aspartate aminotransferase [Enzym_value (IU/L)',
        'Bilirubin.total [Mass/vol', 'Urea nitrogen [Mass/volume] in Serum _value (mg/dL)', 'BUN/Creatinine ratio', 'C peptide [Mass/volume] in Seru_value (ng/L)', 'Calcium [Mass/volume] in Serum or_value (mEq/L)', 'Carbon dioxide', 'Chloride [Moles/volume] in Serum_value (mEq/L)', 
        'Creatinine [Mass/volume] in Se_value (mg/dL)', 'C reactive protein [Mass/volume] i_value (mg/L)', 'Globulin [Mass/volume] in ', 'Glucose [Mass/volume] in Serum or_value (mg/dL)', 'Hemoglobin A1c/Hemoglobin.total in _value (%)', 'Cholesterol in HDL [Mass/_value (mg/dL)', 
        'Insulin [Units/volume] in Serum o_value (ng/L)', 'Cholesterol in LDL [Mass/_value (mg/dL)', 'Natriuretic peptide.B prohormon_value (pg/mL)', 'Potassium [Moles/volume] in Ser_value (mEq/L)', 'Protein [Mass/volume] in Se_value (g/dL)', 'Sodium [Moles/volume] in Serum or _value (mEq/L)', 
        'Cholesterol [Mass/volum_value (mg/dL)', 'Triglyceride [Mass/volume] _value (mg/dL)', 'Troponin T.cardiac [Mass/volum_value (ng/L)', 'Albumin [Mass/volume] in Ur_value (_g/mL)', 'Creatinine [Mass/volume]_value (_g/mL)', 'Hemoglobin - g/dL', 'Hematocrit\xa0 - %', 'MCH - pg', 'MCHC - g/dL', 
        'MCV - fL', 'Platelets - x10E3/µL', 'Red Blood Cells (RBC) - x10E6/µL', 'RDW - %', 'White Blood Cells (WBC) - x10E3/µL', 'lettera_time', 'memory_trial1', 'memory_trial1_time', 'memory_trial2', 'memory_trial2_time', 'moca_abstraction', 'moca_abstraction_time', 'moca_combined_mis_score', 
        'moca_orientation', 'moca_orientation_time', 'moca_total_score', 'naming', 'naming_time', 'repetition', 'repetition_time', 'subtraction', 'subtraction_time', 'trails_visuospatial_executive', 'trails_visuospatial_executive_time', 'Waist Circumference (cm)', 'Weight (kilograms)', 
        'Waist to Hip Ratio (WHR)', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T'])
    
    #length of cat_cols
    CAT_DIM: int = 35
    #target label for classification evaluation 
    LABEL: str = "Type 2 Diabetes"

    #-----MODEL PARAMETERS-----#
    DEVICE: str = 'cpu'
    BATCH: int = 64 
    NOISE_DIM: int = 128  
    GRADIENT_PENALTY: int = 5
    G_LR: float = 1e-4
    G_H: int = 128 
    D_LR: float = 1e-4
    D_H: int = 128
    DISC_STEPS: int = 3
    NUM_SAMPLES: int = 10_518
    
    #PPO SPECIFIC  
    USE_TANH: bool = True  
    #number of times to iterate over PPO training loop (gen steps)
    PPO_EPOCHS: int = 5
    MEAN_PENALTY_SCALE: float = 0.2 #0.1
    VF_COEF: float = 0.5 #default in PPO, pretty standard
    CLIP_EPS: float = 0.1 #can also try 0.2 
    ENT_BETA: float = 1e-3 #can also try 0.01 
    #simple clip on tanh to avoid bugs 
    EPS: float = 1e-6 

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=2))
    def override(self, **kwargs):
        return replace(self, **kwargs)


