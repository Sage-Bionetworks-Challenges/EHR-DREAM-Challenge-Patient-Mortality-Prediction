## graphing
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns


## other
import pandas as pd 
import numpy as np
import argparse
import random
import math
import os
import json
from datetime import datetime as dt


## Custom Modules
from predictions import get_goldstandard
from predictions import get_top_submissions_list
from predictions import collect_predictions_via_team

from patient_information import CHALLENGE_DATA, demographic_boxplot_distributions
from patient_information import race_submission_data
from patient_information import generate_bayes_factors

## sklearn packages
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve


## Delong Test
from roc_comparison.compare_auc_delong_xu import delong_roc_test

## pull in the model output directory
MODEL_OUTPUT = json.load(open("config.json"))["MODEL_OUTPUT"]

## set the colors for each team.
TEAM_PALETTE = {
    'Medil': '#A9A9A9',
    'ivanbrugere': '#FF4500',
    'AI4Life': '#A9A9A9',
    'UW-biostat': '#20B2AA',
    'moore': '#A9A9A9',
    'DMIS_EHR': '#0080FF',
    'markosbotsaris': '#A9A9A9',
    'shikhar-omar': '#A9A9A9',
    'Georgetown - ESAC': '#A9A9A9',
    'LCSB_LUX': '#A9A9A9',
    'avati': '#A9A9A9',
    'AMbeRland': '#6666FF',
    'ProActa': '#B22222',
    'ESACinc Team': '#A9A9A9',
    'ultramangod671': '#A9A9A9',
    'chk': '#A9A9A9',
    'tgaudelet': '#A9A9A9',
    'yachee': '#A9A9A9',
    'paulperry': '#A9A9A9',
    'QiaoHezhe': '#A9A9A9',
    'Kkadri': '#A9A9A9',
    'strucka': '#A9A9A9',
    'O__O': '#A9A9A9',
    'PnP_India': '#A9A9A9',
    'HELM': '#A9A9A9',
    'arppa99100': '#A9A9A9'
}


## generates the delong p values by comparing 
def generate_p_values(submissions):
    gold = get_goldstandard()

    preds = list(submissions["Team"])
    
    preds_pvalues = []
    for i in range(len(preds)-1):
        
        pred1 = preds[i]
        predictions1 = pd.read_csv(f"{MODEL_OUTPUT}/{pred1}/predictions.csv")

        pred2 = preds[i+1]
        predictions2 = pd.read_csv(f"{MODEL_OUTPUT}/{pred2}/predictions.csv")

        all_preds = gold.merge(predictions1, on="person_id", how="left")
        all_preds = all_preds.merge(predictions2, on="person_id", how="left", suffixes=("_1","_2"))
        
        g = np.array(all_preds["status"])
        p1 = np.array(all_preds["score_1"])
        p2 = np.array(all_preds["score_2"])

        scoring = delong_roc_test(g, p1, p2)
        temp = {
            "submission_high":pred1, 
            "submission_low":pred2, 
            "Delong Test p value":float(math.e)**float(scoring[0][0])
        }
        preds_pvalues.append(temp)
    preds_pvalues = pd.DataFrame(preds_pvalues)
    return preds_pvalues


## Generates the callibration curves for all top submissions, broken down by the input patient characteristic.
def calibration_curves(col_name):
    gold = get_goldstandard()
    top = get_top_submissions_list()

    predictions = collect_predictions_via_team(top)
    predictions = predictions.merge(gold, on="person_id", how="left")

    submissions = list(top["UW Queue Id"])
    teams = list(top["Team"])

    calibrations = pd.DataFrame()
    count = 0
    for sub in submissions:
        #print (sub, int(count/(len(submissions)/4)))
        prob_true, prob_pred = calibration_curve(predictions["status"], predictions[f"{sub}"], n_bins=10, strategy="uniform")
        temp = {
            "Team": [teams[count] for i in range(len(prob_true))],
            "Fraction of Positives": prob_true,
            "Mean Predicted Probability": prob_pred,
            "Group": int(count/(len(submissions)/4)),
        }
        temp = pd.DataFrame(temp)
        calibrations = pd.concat([calibrations, temp])
        count += 1
    calibrations = pd.DataFrame(calibrations)
    calibrations.dropna(inplace=True)
    
    g = sns.FacetGrid(calibrations, col="Team", hue="Team", sharey=True, sharex=True, xlim=(0.0, 1.0), ylim=(0.0, 1.0), col_wrap=8, aspect=0.9)

    g.map(sns.lineplot, "Mean Predicted Probability", "Fraction of Positives", linewidth=3)
    g.map(sns.scatterplot, "Mean Predicted Probability", "Fraction of Positives")
    g.map(sns.lineplot, x=[0, 1], y=[0, 1], linewidth=1, dashes=True)

    g.set_titles(f"{col_name}", size=15)
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Calibration Curves of top submissions order by rank', fontsize=30)
        
    plt.savefig("plots/calibration_curves_top_submissions.png", bbox_inches="tight")


## Collects the Area Under the Precision Recall Curve for a given prediction file.
def get_AUPR(pred_path, gold_path):

    try:
        preds = pd.read_csv(pred_path)
    except FileNotFoundError:
        return "-"

    try:
        gold  = pd.read_csv(gold_path)
    except FileNotFoundError:
        return "-"

    data = gold.merge(preds, on="person_id", how="inner")

    precision, recall, thresholds = precision_recall_curve(data["status"], data["score"], pos_label=1)
    pr_auc = auc(recall, precision)
    prauc_score = round(pr_auc, 3)

    return prauc_score


## Collects the Area Under the Receiver Operator Curve for a given prediction file.
def get_AUROC(pred_path, gold_path):

    #print (pred_path)

    try:
        preds = pd.read_csv(pred_path)
    except FileNotFoundError:
        return "-"

    try:
        gold  = pd.read_csv(gold_path)
    except FileNotFoundError:
        return "-"
    
    data = gold.merge(preds, on="person_id", how="inner")
    #print (data)

    fpr, tpr, thresholds = roc_curve(data["status"], data["score"], pos_label=1)
    auroc_score = round(auc(fpr, tpr), 3)

    return auroc_score

## Generate 95% confidence infervals for a given model prediction. CIs are generated using bootstrapping.
def AUROC_CI(pred_path, gold_path, output_path, num_bootstraps=1000):

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    if os.path.isfile(f"{output_path}/full_{num_bootstraps}_bootstraps.csv"):
        all_aurocs = pd.read_csv(f"{output_path}/full_{num_bootstraps}_bootstraps.csv")
    else:
        try:
            preds = pd.read_csv(pred_path)
        except FileNotFoundError:
            return "-"

        try:
            gold  = pd.read_csv(gold_path)
        except FileNotFoundError:
            return "-"
        
        data = gold.merge(preds, on="person_id", how="inner")
        all_aurocs = []
        for n in range(num_bootstraps):
            sampled_data = data.sample(n=len(data), replace=True)

            fpr, tpr, thresholds = roc_curve(sampled_data["status"], sampled_data["score"], pos_label=1)
            auroc_score = auc(fpr, tpr)
            all_aurocs.append(auroc_score)

        pd.Series(all_aurocs).to_csv(f"{output_path}/full_{num_bootstraps}_bootstraps.csv", header=["Scores"], index=False)

    alpha = 0.95

    p = ((1.0-alpha)/2.0) * 100
    lower = round(np.percentile(all_aurocs, p), 3)

    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = round(np.percentile(all_aurocs, p), 3)

    return f"{lower}-{upper}"


## Generates the different performances of the different teams across the three main datasets used in the challenge.
def generating_paper_score_table():

    top_subs = get_top_submissions_list()

    paper_leaderboard = []
    
    for team in top_subs["Team"]:
        temp = {}
        submission = top_subs[top_subs["Team"]==team]["UW Queue Id"].values[0]
        temp["Team"] = team
        print (team)

        if True:
            ## Leaderboard Evaluation Scores
            leaderboard  = f"/data/common/data/DREAM/data/model_reruns/{team}/ensemble/output/evaluation/predictions.csv"
            goldstandard = f"/data/common/data/DREAM/data/model_reruns/goldstandards/ensemble/goldstandard_eval.csv"
            temp["Leaderboard Phase AUROC"] = get_AUROC(leaderboard, goldstandard)
            
            bootstrap_output = f"/data/common/data/DREAM/permutations/{team}/ensemble_evaluation"
            temp["Leaderboard AUROC 95% CI"] = AUROC_CI(leaderboard, goldstandard, bootstrap_output)
            
            temp["Leaderboard Phase AUPR"] = get_AUPR(leaderboard, goldstandard)


            ## Leaderboard Validation Scores
            validation   = f"/data/common/data/DREAM/data/model_reruns/{team}/ensemble/output/validation/predictions.csv"
            goldstandard = f"/data/common/data/DREAM/data/model_reruns/goldstandards/ensemble/goldstandard_valid.csv"
            temp["Validation Phase AUROC"] = get_AUROC(validation, goldstandard)

            bootstrap_output = f"/data/common/data/DREAM/permutations/{team}/validation"
            temp["Validation AUROC 95% CI"] = AUROC_CI(validation, goldstandard, bootstrap_output)

            temp["Validation Phase AUPR"] = get_AUPR(validation, goldstandard)
            

            ## Post Evaluation 80/20 Resplit Scores
            resplit      = f"/data/common/data/DREAM/data/model_reruns/{team}/two_split/output/evaluation/predictions.csv"
            goldstandard = f"/data/common/data/DREAM/data/model_reruns/goldstandards/two_split_data/goldstandard_eval.csv"
            temp["Resplit Phase AUROC"] = get_AUROC(resplit, goldstandard)

            bootstrap_output = f"/data/common/data/DREAM/permutations/{team}/two_split"
            temp["Resplit AUROC 95% CI"] = AUROC_CI(resplit, goldstandard, bootstrap_output)
            temp["Resplit Phase AUPR"] = get_AUPR(resplit, goldstandard)
            
            print (temp)

        paper_leaderboard.append(temp)

    paper_leaderboard = pd.DataFrame(paper_leaderboard)
    #print (paper_leaderboard)
    paper_leaderboard.to_csv("data/Table 2.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the code used to evaluate the mortality prediction models from the EHR DREAM Challenge: Mortality Prediction.")
    parser.add_argument('Analysis', metavar="Analysis (table2, figure3, supp_figure5, supp_figure6, supp_figure7, supp_figure8, supp_figure9)", type=str, nargs=1, choices=['table2', 'figure3', 'supp_figure5', 'supp_figure6', 'supp_figure7', 'supp_figure8', 'supp_figure9'], 
        help="Choose the analysis to perform. 'table2' will generate the data for table2, 'figure3' will generate the data and figure for Figure 3, 'supp_figure4' will generate the data and figure for Supplemental Figure 4, etc.")
    
    args = parser.parse_args()
    print (args.Analysis)
    if args.Analysis[0] == "table2":
        print ("Because this function was specific to the Mortality Prediction challenge evaluation, this function won't run unless you go into the 'generating_paper_score_table() function and designate the sources of the three different datasets.'")
        #generating_paper_score_table()

    if args.Analysis[0] == "figure3":
        cur_categories = ["Race"]
        demographic_boxplot_distributions(cur_categories)
        generate_bayes_factors()

    if args.Analysis[0] == "supp_figure5":
        cur_categories = ["Ethnicity", "Gender"]
        demographic_boxplot_distributions(cur_categories)

    if args.Analysis[0] == "supp_figure6":
        top_subs = get_top_submissions_list()
        subs, data, cats, col = race_submission_data(top_subs)
        calibration_curves(subs, data, cats, col)

    if args.Analysis[0] == "supp_figure7":
        cur_categories = ["Age"]
        demographic_boxplot_distributions(cur_categories)

    if args.Analysis[0] == "supp_figure8":
        cur_categories = ["prior_visits"]
        demographic_boxplot_distributions(cur_categories)

    if args.Analysis[0] == "supp_figure9":
        cur_categories = ["visit_type"]
        demographic_boxplot_distributions(cur_categories)
