# EHR DREAM Challenge - Patient Mortality Prediction
This is the repository for the code used to generate the results of the analysis of the EHR DREAM Challenge: Patient Mortality Prediction.

## Installation
Add the packages into a conda environment.
```
conda create --name dream --file requirements.txt
```
Clone the following github repos into the main directory.
```
git clone https://github.com/UWMooneyLab/DxCodeHandler.git
git clone https://github.com/yandexdataschool/roc_comparison.git
```
## Configurations
In order to run this code, you'll need three things: 
1. The directory that contains the patient-level OMOP data that the models made their predictions on.
2. The directory in which the models output their predictions. Each model should be outputing the predictions into a folder named after their team. (e.g. /path/to/output/directory/team)
3. The gold standard file which is a csv file that has two columns, "person_id" and "status", and the status is the true outcome (180 day mortality status) of each patient in the OMOP table person.

Fill in the `config.json` file with the paths to the directories.

Once these are complete, you can run the following options to reproduce figures and tables from the paper.

## Analysis

### Table 2 Code
Generate table 2. This function will show the message `Because this function was specific to the Mortality Prediction challenge evaluation, this function won't run unless you go into the 'generating_paper_score_table() function and designate the sources of the three different datasets.` Because the function uses three different datasets and looks at three sets of mortality predictions from each team, in order to use this function, in-function customization is required.  
```
python MainEval.py table2
```
The Delong P values from Table 2 were generated using the `generate_p_values` function in `MainEval.py`.

### Figure 2
Figure 2 was generated using an R script. You can see the `Figure 2.R`. This script ingests the `data/Table_2_Data.csv` dataset.

### Figure 3
Breaks down the model predictions by Race, carries out `N` number of bootstraps (specified in the `config.json` file) to generate a distribution, and generates the box plots seen in Figure 3. Will also calculate the Bayes Factors found in Supplemental Table 2.
```
python MainEval.py figure3
```

### Figure 4
Figure 4 was generated using the `ICD_Impact.py` script to calculate the z scores associated with each ICD10 code. The `ICD_Impact_Analysis.py` script will take the data output from `ICD_Impact.py` and generate the data used in the lineplot for Figure 4. The data from the challenge is availabe at `data/Figure_4_Data.csv` as well as in Supplemental Table 3 in the paper.
The tree plot in Figure 4b was generated using the VEDx tool. https://github.com/UWMooneyLab/VEDx

### Supplemental Figure 5
Breaks down the model predictions by Ethnicity and Gender, carries out `N` number of bootstraps (specified in the `config.json` file) to generate a distribution, and generates the box plots seen in Supplemental Figure 5.
```
python MainEval.py supp_figure5
```

### Supplemental Figure 6
Generates calibration curves broken down by Race for each model's predictions.
```
python MainEval.py supp_figure6
```

### Supplemental Figure 7
Breaks down the model predictions by Age of the patient, carries out `N` number of bootstraps (specified in the `config.json` file) to generate a distribution, and generates the box plots seen in Supplemental Figure 7.
```
python MainEval.py supp_figure7
```

### Supplemental Figure 8
Breaks down the model predictions based on the number of visits prior the latest visit of each patient, carries out `N` number of bootstraps (specified in the `config.json` file) to generate a distribution, and generates the box plots seen in Supplemental Figure 8.
```
python MainEval.py supp_figure8
```

### Supplemental Figure 9
Breaks down the model predictions based on the visit type (ER, Outpatient, Inpatient) of the latest visit of each patient, carries out `N` number of bootstraps (specified in the `config.json` file) to generate a distribution, and generates the box plots seen in Supplemental Figure 9.
```
python MainEval.py supp_figure9
```