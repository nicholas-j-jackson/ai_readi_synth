# Synthetic Fundus Photography Dataset of Type 2 Diabetes Derived from the AI-READI Project

## Version number

1.0.0

## Publication date

2025-XX-XX

## Description of the dataset

This dataset contains 3 main components: (1) A synthetic tabular dataset containing XXXX person-days of clinical data, wearable data, ekgs, and XXXXX. (2) A dataset of 43,520 synthetically generated retinal fundus images from 6 different imaging devices. (3) A dataset of XXX synthetically generated OCT scans from 3 different imaging devices. The tabular dataset is stored in OMOP format alongisde several additional tables for data derived from EKGs. The OCT and fundus datasets are stored in DICOM files and the accompanying labels (for age-related macular degeneration, diabetic retinopathy, and glaucoma) are available in OMOP format. A complete description is provided in the "Data Standards" section below. The data in this dataset contain no protected health information (PHI). Information related to the sex and race/ethnicity of the participants as well as medication used has also been removed.

A detailed description of the real dataset is available in the AI-READI documentation for v2.0.0 of the dataset at [docs.aireadi.org](https://docs.aireadi.org/). Synthetic data have been re-formatted to match the structure of the real dataset and enable interoperability of code developed on the synthetic dataset.

## Protocol

The protocol followed for collecting the data can be found in the AI-READI documentation for v2.0.0 of the dataset at [docs.aireadi.org](https://docs.aireadi.org/).

## Dataset access/restrictions

Accessing the dataset requires several steps, including:

- Login in through a verified ID system
- Agreeing to use the data only for type 2 diabetes related research.
- Agreeing to the license terms which set certain restrictions and obligations for data usage (see "License" section below).

## Data standards followed

This dataset is organized following the [Clinical Dataset Structure (CDS) v0.1.1](https://cds-specification.readthedocs.io/en/v0.1.1/). We refer to the CDS documentation for more details. Briefly, data is organized at the root level into one directory per datatype (c.f. Table below). Within each datatype folder, there is one folder per modality. Within each modality folder, there is one folder per device used to collect that modality. Within each device folder, there is one folder per participant. Each datatype, modality, and device folder is named using a name that best defines it. Each participant folder is named after the participant's ID number used in the study. For each datatype, the data files follow the standards listed in the Table below. More details are available in the dataset_structure_description.json metadata file included in this dataset.

| Datatype directory name   | Description                                                                                                                                                                                      | File format standard followed                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| retinal_photography       | This directory contains retinal photography data, which are 2D images. They are also referred to as fundus photography. These images were synthetically generated and each correspond to a new synthetic participant.                                                                          | [Digital Imaging and Communications in Medicine (DICOM)](http://medical.nema.org/)                                                                                |
| clinical_data             | This directory contains clinical data collected through REDCap. Each CSV file in this directory is a one-to-one mapping to the OMOP CDM tables. The condition_occurrence and measurement file are the files for which synthetic data are available.                                                | [Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM)](https://ohdsi.github.io/TheBookOfOhdsi)                                               |                                                                |

## Resources

All of our data files are in formats that are accessible with free software commonly used for such data types so no specific software is required. Some useful resources related to this dataset are listed below:

- Documentation of the dataset: [docs.aireadi.org](https://docs.aireadi.org/) (see 'Dataset v2.0.0' which corresponds to this synthetic dataset)
- AI-READI project website: [aireadi.org](https://aireadi.org/)
- Zenodo community of the AI-READI project: [zenodo.org/communities/aireadi](https://zenodo.org/communities/aireadi)
- GitHub organization of the AI-READI project: [github.com/AI-READI](https://github.com/AI-READI)

### Changes between versions of the dataset

Changes between the current version of the dataset and the previous one are provided in details in the CHANGELOG file included in the dataset (also visible at docs.aireadi.org). A summary of the major changes is provided in the table below.

| Dataset      | v1.0.0 pilot    | year 2 data        | v2.0.0 main study  | v1.0.0 synthetic data |
| ------------ | --------------- | ------------------ | ------------------ | --------------------- |
| Participants | 204             | 863                | 1067               | 43520                 |
| Data types   | 15+ data types  | +1 image device    | 15+ data types     | retinal fundus images |
| Processing   | custom / ad hoc | automated + custom | automated + custom | annotated + custom    |
| Release date | 5/3/2024        | included in v2.0.0 | 11/8/2024          | XXXXXX                |

## License

XXXXXXXXXXXXX

## How to cite

If you use this dataset for any purpose, please cite the resources specified in the AI-READI documentation for version 2.0.0 of the dataset at https://docs.aireadi.org.

## Contact

For any questions, suggestions, or feedback related to this dataset, please go to https://aireadi.org/contact.

## Acknowledgement

The AI-READI project is supported by NIH grant [1OT2OD032644](https://reporter.nih.gov/search/1ADgncihCk6fdMRJdCnBjg/project-details/10471118) through the NIH Bridge2AI Common Fund program.
