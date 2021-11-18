from typing import no_type_check
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

from DxCodeHandler.ICD10 import ICD10
icd = ICD10(errorHandle="NoDx")

import os

## graphing
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import dexplot as dx


def get_current_ICD_impact_scores():
    data = pd.read_csv("ICD_impact_z_results.csv")
    return data

def spread(x):
    return np.mean([abs(np.mean(x) - np.max(x)), abs(np.mean(x) - np.min(x))])

if __name__ == "__main__":

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

    scores = get_current_ICD_impact_scores()
    
    score_metrics = pd.DataFrame(scores.groupby(["Diagnosis Code", "inclusion cohort size"]).agg({
        "Z score": ["min", "max", "mean", spread]
    })).reset_index()
    score_metrics.columns = ["Diagnosis Code", "Inclusion Cohort Size", "Min", "Max", "Mean", "Spread"]
    
    score_metrics["description"] = score_metrics["Diagnosis Code"].apply(lambda x: icd.description(x))
    score_metrics["level"] = score_metrics["Diagnosis Code"].apply(lambda x: icd.depth(x))
    print (score_metrics.sort_values("Mean", ascending=False).head(50))
    score_metrics.sort_values("Mean", ascending=False).head(50).to_csv("top_50_ranked_codes.csv", index=False)

    score_metrics.columns = ["codes", "Inclusion Cohort Size", "Min", "Max", "score", "color", "description", 'depth']
    score_metrics = score_metrics[score_metrics["depth"] <= 4]
    score_metrics.sort_values("score", ascending=False).to_csv("vedx_ranked_codes_w_color.csv", index=False)


    
    scores["depth"] = scores["Diagnosis Code"].apply(lambda x: icd.depth(x))
    scores = scores[scores["depth"]==1]
    print (scores)

    cumulative_measure = "Median Z Score"
    TEAM_PALETTE[cumulative_measure] = '#797979'

    if cumulative_measure == "Mean":
        agg_scores = pd.DataFrame(scores.groupby("Diagnosis Code")[["full_roc_auc","exclusion_roc_auc","Z score"]].mean()).reset_index()
    else:
        agg_scores = pd.DataFrame(scores.groupby("Diagnosis Code")[["full_roc_auc","exclusion_roc_auc","Z score"]].median()).reset_index()
    
    agg_scores["Team"] = cumulative_measure
    agg_scores["depth"] = agg_scores["Diagnosis Code"].apply(lambda x: icd.depth(x))

    scores = pd.concat([scores, agg_scores])

    scores = scores.sort_values(["Z score", "Team"], ascending=False)

    short_descriptions = {
        'A00-B99': 'Infectious Diseases',
        "C00-D49": "Neoplasms",
        "D50-D89": "Diseases of the blood and immune mechanisms",
        "E00-E89": "Metabolic diseases",
        "F01-F99": "Mental disorders",
        "G00-G99": "Diseases of the nervous system",
        "H00-H59": "Diseases of the eye",
        "H60-H95": "Diseases of the ear",
        "I00-I99": "Diseases of the circulatory system",
        "J00-J99": "Diseases of the respiratory system",
        "K00-K95": "Diseases of the digestive system",
        "L00-L99": "Diseases of the skin",
        "M00-M99": "Diseases of the musculoskeletal system",
        "N00-N99": "Diseases of the genitourinary system",
        "O00-O9A": "Pregnancy, childbirth and the puerperium",
        "P00-P96": "Conditions from the perinatal period",
        "Q00-Q99": "Congenital deformations",
        "R00-R99": "Abnormal clinical findings",
        "S00-T88": "Injury from external causes",
        "Z00-Z99": "Factors influencing health status",
    }

    score_metrics["Diagnosis Category"] = score_metrics["codes"].map(short_descriptions)

    teams = ['UW-biostat','ivanbrugere','ProActa','AMbeRland','DMIS_EHR','PnP_India','ultramangod671','HELM','AI4Life','Georgetown - ESAC', cumulative_measure]
    score_metrics = score_metrics.sort_values("score", ascending=False)
    ordered_categories = list(score_metrics[score_metrics["depth"]==1]["codes"].drop_duplicates())
    ordered_categories = list(score_metrics[score_metrics["depth"]==1].sort_values("codes")["codes"].drop_duplicates())

    number_of_teams = len(teams)
    top_team_count = 3
    top_teams = teams[:top_team_count]
    bottom_teams = [t for t in teams[top_team_count:] if t != cumulative_measure]

    start = 0.35
    stop = 0.0
    step = -(start-stop)/(len(bottom_teams))
    print (start, stop, step)
    top_team_alphas = [1.0,0.75,0.5]
    new_alphas = [a for a in np.arange(start, stop, step)]
    new_alphas = [0.5 for a in np.arange(start, stop, step)]
    print (new_alphas)

    scores[["Team", "Diagnosis Code", "full_roc_auc", "Z score"]].to_csv("figure_4_lineplot.csv", index=False)

    a4_dims = (60, 12)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.pointplot(ax=ax, data=scores[scores["Team"].isin(top_teams)], y="Z score", x="Diagnosis Code", scale=2.5, zorder=10, order=ordered_categories, hue="Team", markers=['o','d','x'], hue_order=top_teams, palette=TEAM_PALETTE, ci=None) # )
    g = sns.pointplot(ax=ax, data=scores[scores["Team"].isin(bottom_teams)], y="Z score", x="Diagnosis Code", scale=0.95, order=ordered_categories, hue="Team", hue_order=bottom_teams, palette=TEAM_PALETTE, ci=None) # )
    sns.pointplot(ax=ax, data=scores[scores["Team"] == cumulative_measure], y="Z score", x="Diagnosis Code", scale=2, zorder=2, order=ordered_categories, hue="Team", alpha=0.6, markers='|', linestyles='dashdot', palette=TEAM_PALETTE, ci=None) # )
    
    #g = sns.stripplot(data=scores[scores["Team"].isin(teams.tail(number_of_teams-top_teams))], y="Z score", x="Diagnosis Code", order=ordered_categories, hue="Team", hue_order=teams.tail(number_of_teams-top_teams), palette=TEAM_PALETTE, dodge=False)# )

    for i in range(top_team_count):
        #print(g.collections[top_team_count:][i:i+1])
        plt.setp(g.collections[:top_team_count][i:i+1], zorder=top_team_count+5-i, alpha=top_team_alphas[i]) #for the markers
        plt.setp(g.lines[:top_team_count][i:i+1], zorder=top_team_count+5-i, alpha=top_team_alphas[i])       #for the lines

    for i in range(len(new_alphas)):
        #print(g.collections[top_team_count:][i:i+1])
        if teams[top_team_count:][i] == cumulative_measure:
            plt.setp(g.collections[top_team_count:][i:i+1], alpha=0.6, zorder=2) #for the markers
            plt.setp(g.lines[top_team_count:][i:i+1], zorder=2)
        else:
            plt.setp(g.collections[top_team_count:][i:i+1], alpha=new_alphas[i], zorder=1) #for the markers
            plt.setp(g.lines[top_team_count:][i:i+1], alpha=new_alphas[i], zorder=1)       #for the lines
    
    
    g.set(xticklabels=[])
    plt.yticks(fontsize=45)
    plt.xlabel(None, fontsize=40)
    plt.ylabel('Z Score', fontsize=45)

    g.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0., fontsize=40, markerscale=2)

    plt.savefig("stripplot_z-scores.png", bbox_inches="tight")