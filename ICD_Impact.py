import pandas as pd
from MainEval import MODEL_OUTPUT
import numpy as np
from patient_information import NUM_BOOTSTRAPS
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

from predictions import get_goldstandard
from predictions import get_top_submissions_list

import os
import json

## graphing
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_OUTPUT = json.load(open("config.json"))["MODEL_OUTPUT"]
CHALLENGE_DATA = json.load(open("config.json"))["CHALLENGE_DATA"]
NUM_BOOTSTRAPS = json.load(open("config.json"))["NUM_BOOTSTRAPS"]


def get_main_prediction_file(team):
    path = f"{MODEL_OUTPUT}/{team}/predictions.csv"
    try:
        predictions = pd.read_csv(path)
    except FileNotFoundError:
        print (f"{team} has no predictions")
        predictions = pd.DataFrame()
    return predictions 


def get_roc_auc(team, data):

    try:
        fpr, tpr, thresholds = roc_curve(data["status"], data[f"{team}"], pos_label=1)
        roc_auc = auc(fpr, tpr)
        auroc_score = float(round(roc_auc, 5))
    except ValueError:
        auroc_score = 1.0
    return auroc_score


def hanley_mcneil_standard_error(auc, goldstandard):

    Q1 = auc/(2-auc)
    Q2 = (((2*auc)**2))/(1+auc)

    N_tp = np.sum(goldstandard["status"])
    #print ("N TP", N_tp)

    N_tn = len(goldstandard) - N_tp
    #print ("N TN", N_tn)

    top = auc*(1-auc) + (N_tp-1)*(Q1-(auc**2)) + (N_tn-1)*(Q2-(auc**2))
    bottom = N_tp*N_tn

    SE = np.sqrt(top/bottom)
    return SE


def get_z_score(auc_a, auc_b, sd_a, sd_b, N_a, N_b):
    Z = (auc_a - auc_b) / (np.sqrt((sd_a)**2 / N_a + (sd_b)**2 / N_b))
    return Z


def get_ICD_impact_z_score(team, code, full_auc, exclusion_auc, full_se, N_full, N_exclusion, goldstandard):
    temp = {}

    temp["Team"] = team
    temp["Diagnosis Code"] = code
    temp["full_roc_auc"] = full_auc
    temp["exclusion_roc_auc"] = exclusion_auc

    SE_exclusion = hanley_mcneil_standard_error(exclusion_auc, goldstandard)

    #print (full_auc, full_se, exclusion_auc, SE_exclusion)

    temp["Z score"] = get_z_score(full_auc, exclusion_auc, full_se, SE_exclusion, N_full, N_exclusion)
    temp["exclusion cohort size"] = N_exclusion
    temp["inclusion cohort size"] = N_full - N_exclusion
    return temp


def get_ICD_impact_score(team, code, associated_codes):
    temp = {}

    inclusion_cohort = associated_codes[associated_codes["Diagnosis Code"]==code][["person_id", "status", "score"]].drop_duplicates()
    exclusion_cohort = associated_codes[~associated_codes["person_id"].isin(set(inclusion_cohort["person_id"]))][["person_id", "status", "score"]].drop_duplicates()

    temp["Team"] = team
    temp["Diagnosis Code"] = code
    temp["inclusion_roc_auc"] = get_roc_auc("score", inclusion_cohort)
    temp["exclusion_roc_auc"] = get_roc_auc("score", exclusion_cohort)
    temp["inclusion cohort size"] = len(inclusion_cohort)
    temp["exclusion cohort size"] = len(exclusion_cohort)
    
    return temp


def get_submissions():
    teams = pd.read_csv("top_submissions.csv")
    return teams


def create_ICD_codes_dates():
    from DxCodeHandler.ICD9 import ICD9
    from DxCodeHandler.ICD10 import ICD10
    from DxCodeHandler.Converter import Converter

    icd9 = ICD9()
    icd10 = ICD10()
    conv = Converter()

    condition_data = pd.read_csv(f"{CHALLENGE_DATA}/condition_occurrence.csv", usecols=["person_id", "condition_start_date", "condition_source_value"], parse_dates=["condition_start_date"])
    visit_data = pd.read_csv(f"{CHALLENGE_DATA}/visit_occurrence.csv", usecols=["person_id", "visit_start_date"], parse_dates=["visit_start_date"])
    max_visit = pd.DataFrame(visit_data.groupby("person_id")["visit_start_date"].max().reset_index())
    max_visit["beginning_window"] = max_visit["visit_start_date"].apply(lambda x: x - timedelta(days=180))

    data = condition_data.merge(max_visit, on="person_id", how="left")
    data = data[data["condition_start_date"]>=data["beginning_window"]]

    unique_codes = data[["condition_source_value"]].drop_duplicates()

    unique_codes["ICD9"] = unique_codes["condition_source_value"].apply(lambda x: icd9.isCode(x))
    print ("ICD 9 unique code count", len(unique_codes[unique_codes["ICD9"]]))

    unique_codes["ICD10"] = unique_codes["condition_source_value"].apply(lambda x: icd10.isCode(x))
    print ("ICD 10 unique code count", len(unique_codes[unique_codes["ICD10"]]))

    ICD9_codes = unique_codes[(unique_codes["ICD9"])&(unique_codes["ICD10"]==False)]

    def con_9_2_10(code):
        try:
            converted_code = conv.convert_9_10(code)
            if len(converted_code) == 1:
                return converted_code
            else:
                return converted_code

        except Exception:
            return "No Conversion"
    
    ICD9_codes["converted"] = ICD9_codes["condition_source_value"].apply(lambda x: con_9_2_10(x))
    
    clean_conversions = ICD9_codes[(ICD9_codes["converted"]!="No Conversion")]
    clean_conversions = clean_conversions.explode("converted")
    clean_conversions = clean_conversions[["condition_source_value", "converted"]]
    clean_conversions.columns = ["ICD9", "ICD10"]

    data = data.merge(clean_conversions, left_on="condition_source_value", right_on="ICD9", how="left")
    data["Diagnosis Code"] = data["ICD10"].combine_first(data["condition_source_value"])
    data = data[["person_id", "Diagnosis Code"]]
    data["Diagnosis Code"] = data["Diagnosis Code"].apply(lambda x: icd10.ancestors(x))
    data = data.explode("Diagnosis Code")
    data["Depth"] = data["Diagnosis Code"].apply(lambda x: icd10.depth(x))
    print (data)

    gold = get_goldstandard()
    data = gold.merge(data, on="person_id", how="inner")
    data = data[["person_id", "Diagnosis Code", "Depth", "status"]]
    return data


def get_current_ICD_impact_scores():
    data = pd.read_csv("data/ICD_impact_results.csv")
    return data


def get_associated_codes():
    if os.path.isfile("data/associated_codes_180_days.csv"):
        associated_codes = pd.read_csv("data/associated_codes_180_days.csv")
    else:
        associated_codes = create_ICD_codes_dates()
        associated_codes.to_csv("data/associated_codes_180_days.csv", index=False)
    return associated_codes


def build_ICD_impact_table_to_depth(teams, depth=3):

    associated_codes = get_associated_codes()

    code_counts = pd.DataFrame(associated_codes[["person_id", "Diagnosis Code"]].drop_duplicates().groupby("Diagnosis Code")["person_id"].count()).reset_index()
    codes = code_counts[code_counts["person_id"]>100]["Diagnosis Code"]
    

    if os.path.isfile("data/ICD_impact_z_results.csv"):
        PREVIOUS_RESULTS = pd.read_csv("data/ICD_impact_z_results.csv")
    else:
        PREVIOUS_RESULTS = pd.DataFrame()
    
    try:
        codes = codes[~codes["Diagnosis Code"].isin(set(PREVIOUS_RESULTS["Diagnosis Code"]))]
    except KeyError:
        pass

    print ("Number of codes", len(codes))

    RESULTS = []

    team_full_results = {}
    all_associated_codes = pd.DataFrame()

    #teams = teams[:3]

    goldstandard = associated_codes[["person_id", "status"]].drop_duplicates()

    for team in teams:
        print (team, flush=True)
        team_full_results[team] = {}

        predictions  = get_main_prediction_file(team)
        
        associated_codes = associated_codes.merge(predictions, on="person_id", how="left")
        predictions.columns = ["person_id",f"{team}"]

        if len(all_associated_codes) == 0:
            all_associated_codes = associated_codes.merge(predictions, on="person_id", how="left")
        else:
            all_associated_codes = all_associated_codes.merge(predictions, on="person_id", how="left")
        
        

        team_full_results[team]["full_auc"] = get_roc_auc("score", associated_codes[["person_id", "status", "score"]].drop_duplicates())
        team_full_results[team]["full_se"] = hanley_mcneil_standard_error(team_full_results[team]["full_auc"], goldstandard)
        team_full_results[team]["N_full"] = len(goldstandard)

        associated_codes.drop("score", inplace=True, axis=1)
    all_associated_codes = pd.DataFrame(all_associated_codes.drop_duplicates())
    
    count = 0
    print (f"Number of codes {len(codes)}")
    for code in codes:
        results = []
        print (code)
        count += 1
        if count > 1000000000000000:
            exit()
        else:
        
            print ("Building exclusion cohort")
            #print (all_associated_codes)
            inclusion_cohort = associated_codes[associated_codes["Diagnosis Code"]==code]
            exclusion_cohort = all_associated_codes[~all_associated_codes["person_id"].isin(set(inclusion_cohort["person_id"]))].drop(["Diagnosis Code", "Depth"], axis=1).drop_duplicates()
            
            N_inclusion = code_counts[code_counts["Diagnosis Code"]==code]["person_id"].values[0]
            N_exclusion = team_full_results[team]["N_full"] - N_inclusion

            for team in teams:
                code_exclusion_auc = get_roc_auc(team, exclusion_cohort[["person_id", "status", f"{team}"]])
                temp_results = get_ICD_impact_z_score(team, code, team_full_results[team]["full_auc"], code_exclusion_auc, team_full_results[team]["full_se"], team_full_results[team]["N_full"], N_exclusion, exclusion_cohort)
                results.append(temp_results)
            
            CURRENT_RESULTS = pd.DataFrame(results)
            
            if count > 1:
                CURRENT_RESULTS.to_csv("data/ICD_impact_z_results.csv", mode='a', header=False, index=False)
            else:
                CURRENT_RESULTS.to_csv("data/ICD_impact_z_results.csv", header=True, index=False)
            
            

def get_codes_left_to_calculate(teams):

    run_codes = pd.read_csv("data/ICD_impact_results.csv")#["Diagnosis Code"])
    all_codes = set(pd.read_csv("data/associated_codes_180_days.csv")["Diagnosis Code"].drop_duplicates())

    for team in teams:
        team_ran_codes = set(run_codes[run_codes["Team"]==team]["Diagnosis Code"])
        leftovers = all_codes - team_ran_codes
        print (team, len(leftovers))
    

if __name__ == "__main__":

    teams = get_top_submissions_list()[["Team", "validation_Score"]]

    build_ICD_impact_table_to_depth(teams["Team"], depth=10)
