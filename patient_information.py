## graphing
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

## useful modules
import os
import json
import pandas as pd
import numpy as np
from predictions import get_goldstandard
from predictions import collect_predictions_via_team

#from patient_breakdown import graph_distributions

from patient_categories import cat_descr
from patient_categories import get_cats
from patient_categories import get_version

from predictions import get_top_submissions_list

## Evaluation
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.calibration import calibration_curve

## Delong Test
from roc_comparison.compare_auc_delong_xu import delong_roc_test

NUM_BOOTSTRAPS = json.load(open("config.json"))["NUM_BOOTSTRAPS"]

CHALLENGE_DATA = json.load(open("config.json"))["CHALLENGE_DATA"]

def bootstrap_submission(team, predictions, categories, category_column, version, phase):
    scores = []
    #print (predictions)
    for i in categories:
        print (category_column, i)
        preds = predictions[predictions[category_column] == i]
        preds = preds[["person_id", "status", f"{team}"]]

        if len(preds) == 0:
            pass
        else:
            for boot in range(NUM_BOOTSTRAPS):
                temp = {}
                
                sampled_data = preds.sample(n=len(preds), replace=True)
                #print (sampled_data)

                fpr, tpr, thresholds = roc_curve(sampled_data["status"], sampled_data[f"{team}"], pos_label=1)
                auroc_score = round(auc(fpr, tpr), 4)

                #print (boot, auroc_score)
                
                temp["Team"] = team
                temp[f"{category_column}"] = i
                temp["AUROC"] = float(auroc_score)
                temp["False Positive Rate"] = fpr
                temp["True Positive Rate"] = tpr
                temp["Delong Test"] = delong_roc_test()

                scores.append(temp)

    scores = pd.DataFrame(scores)
    if not os.path.isdir(f"permutations/{team}/{phase}/"):
        os.makedirs(f"permutations/{team}/{phase}/")
    scores.to_csv(f"permutations/{team}/{phase}/{category_column}_{NUM_BOOTSTRAPS}_{version}_bootstraps.csv", index=False)


def intra_method_paired_bayes_factor(submissions, top_subs, predictions, categories, category_column):
    
    print (f"Calculating Bayes Factors for {category_column} with {NUM_BOOTSTRAPS} bootstraps")
    print (top_subs)
    
    if category_column == "Gender":
        categories = ["Female", "Male"]

    if not os.path.isfile(f"data/{category_column}_bayes_factor.csv"):
        significance = []
        for sub in submissions:
            temp = {}
            
            cur_data = predictions[["person_id", "status", f"{sub}", f"{category_column}"]]

            for c in categories:
                for c2 in categories:
                    print (sub, c, c2)
                    category = cur_data[cur_data[f"{category_column}"]==c]
                    category_2 = cur_data[cur_data[f"{category_column}"]==c2]

                    fpr, tpr, thresholds = roc_curve(category["status"], category[f"{sub}"], pos_label=1)
                    auroc_score_cat = round(auc(fpr, tpr), 4)

                    fpr, tpr, thresholds = roc_curve(category_2["status"], category_2[f"{sub}"], pos_label=1)
                    auroc_score_cat_2 = round(auc(fpr, tpr), 4)

                    if auroc_score_cat > auroc_score_cat_2:
                        high_score_category = c
                    else:
                        high_score_category = c2
                    
                    sample_size = min([len(category), len(category_2)])
                    
                    for n in range(NUM_BOOTSTRAPS):
                        
                        sampled_category = category.sample(n=sample_size, replace=True)
                        sampled_category_2 = category_2.sample(n=sample_size, replace=True)

                        fpr, tpr, thresholds = roc_curve(sampled_category["status"], sampled_category[f"{sub}"], pos_label=1)
                        auroc_score_cat = round(auc(fpr, tpr), 4)

                        fpr, tpr, thresholds = roc_curve(sampled_category_2["status"], sampled_category_2[f"{sub}"], pos_label=1)
                        auroc_score_cat_2 = round(auc(fpr, tpr), 4)

                        temp = {
                            "team": top_subs[top_subs["UW Queue Id"]==sub]["Team"].values[0],
                            "submission": sub,
                            "category 1": c,
                            "category 1 score": auroc_score_cat,
                            "category 2": c2,
                            "category 2 score": auroc_score_cat_2,
                            "highest scoring category": high_score_category
                        }
                        significance.append(temp)

        significance = pd.DataFrame(significance)

        significance.to_csv(f"data/{phase}/{category_column}_bayes_factor.csv", index=False)
    else:
        print ("Permutations already built")
        if os.path.isfile(f"data/{category_column}_bayes_factor_calculations.csv"):
            significance = pd.read_csv(f"data/{category_column}_bayes_factor.csv")
            significance["team"] = significance["team"].astype(str)

            significance = significance.merge(top_subs, left_on="team", right_on="UW Queue Id", how="right")

            def count_bayes(x):
                if x["category 1 score"] > x["category 2 score"]:
                    return 1
                else:
                    return 0
            
            def bayes_factor(x, num_bootstraps):
                if num_bootstraps == x:
                    bf = (float(x)/float(num_bootstraps))/(1.0/float(num_bootstraps))
                else:
                    bf = (float(x)/float(num_bootstraps))/(float(num_bootstraps - x)/float(num_bootstraps))
                return bf

            significance["count"] = significance.apply(lambda x: count_bayes(x), axis=1)

            significance = pd.DataFrame(significance.groupby(["Team", "team", "validation_Score", "category 1", "category 2", "highest scoring category"])["count"].sum()).reset_index()
            significance["Bayes Factor"] = significance["count"].apply(lambda x: bayes_factor(x, NUM_BOOTSTRAPS))
            
            bayes_factors = significance.pivot_table(index=["Team", "validation_Score", "category 1"], columns="category 2", values="Bayes Factor").reset_index()
            bayes_factors.to_csv(f"data/{category_column}_bayes_factor_calculations.csv", index=False)
        else:
            bayes_factors = pd.read_csv(f"data/{category_column}_bayes_factor_calculations.csv")

        bayes_factors = bayes_factors[["team", "category 1", "White", "Asian", "Black", "Race Other"]]

        bayes_factors["category 1"] = pd.Categorical(bayes_factors["category 1"], ordered=True, categories=["White", "Asian", "Black", "Race Other"])
        bayes_factors["team"]       = pd.Categorical(bayes_factors["team"], ordered=True, categories=list(top_subs["Team"]))
        
        bayes_factors = bayes_factors.sort_values(["team", "category 1"])

        def create_ceiling(x):
            if x == 0:
                return np.log(float(1/10000))
            else:
                return np.log(x)
        
        bayes_factors["White"] = bayes_factors["White"].apply(lambda x: create_ceiling(x))
        bayes_factors["Asian"] = bayes_factors["Asian"].apply(lambda x: create_ceiling(x))
        bayes_factors["Black"] = bayes_factors["Black"].apply(lambda x: create_ceiling(x))
        bayes_factors["Race Other"] = bayes_factors["Race Other"].apply(lambda x: create_ceiling(x))

        min_value = min([min(bayes_factors["White"]), min(bayes_factors["Asian"]), min(bayes_factors["Black"]), min(bayes_factors["Race Other"])])
        max_value = max([max(bayes_factors["White"]), max(bayes_factors["Asian"]), max(bayes_factors["Black"]), max(bayes_factors["Race Other"])])
        #print (bayes_factors)
        print (min_value, max_value)
        #sns.set(style="white", font="Utopia", rc={'figure.figsize':(4.5,11.27)}) #, font_scale=1.0


        
        bayes_factors.columns = ["Team", "Race", "White", "Asian", "Black", "Other"]
        #g = sns.FacetGrid(bayes_factors, row="Team")
        #g.map(sns.heatmap, data=bayes_factors, center=1, square=True, cmap="coolwarm")

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            data = data.set_index(["Team", "Race"])
            sns.heatmap(data, **kwargs)

        fg = sns.FacetGrid(bayes_factors, row='Team', gridspec_kws={"hspace":0.1})
        fg.map_dataframe(draw_heatmap, center=0, vmin=min_value, vmax=max_value, cmap="coolwarm", linewidths=0.1, cbar=False, square=True, xticklabels=False, yticklabels=False)
        fg.set_titles(row_template="")
        # get figure background color
        facecolor=plt.gcf().get_facecolor()
        for ax in fg.axes.flat:
        #    # set aspect of all axis
            ax.set_aspect('equal','box-forced')
        #    # set background color of axis instance
        #    ax.set_axis_bgcolor(facecolor)

        plt.savefig("plots/bayes_factor_heatmap.png", bbox_inches="tight")


def generate_bayes_factors():
    top_subs = get_top_submissions_list()

    subs, data, cats, col = race_submission_data(top_subs)
    intra_method_paired_bayes_factor(subs, top_subs, data, cats, col)


def graph_boxplots(all_bootstrapped_submissions, category_column):

    print (category_column)
    if category_column == "Age":
        all_bootstrapped_submissions = all_bootstrapped_submissions[all_bootstrapped_submissions[category_column]!="100 +"]
    elif category_column == "Race":
        all_bootstrapped_submissions[category_column] = all_bootstrapped_submissions[category_column].map(
            {"White":"White (n=112,596)", "Asian":"Asian (n=16,145)", "Black":"Black (n=14,154)", "Other":"Other (n=25,812)"}
        )
    elif category_column == "visit_type":
        all_bootstrapped_submissions[category_column] = all_bootstrapped_submissions[category_column].map(
            {"Emergency Room Visit":"Emergency Room Visit (n=151,834)", "Inpatient Visit":"Inpatient Visit (n=57,552)", "Outpatient Visit":"Outpatient Visit (n=1,171,988)"}
        )
    
    colors = {
        "Gender": {"Male":"#2471A3", "Female":"#CD5C5C"},
        "Race": {"White (n=112,596)":"#E74C3C", "Asian (n=16,145)":"#F39C12", "Black (n=14,154)":"#2471A3", "Other (n=25,812)":"#A6ACAF"},
        "Ethnicity": {"Hispanic": "#F39C12", "Not Hispanic": "#2471A3"},
        "Age": {"0 - 17":"#E74C3C", "18 - 34":"#F39C12", "35 - 64":"#2471A3", "65 - 99":"#A6ACAF"}, #, "100 +":"#666666"},
        "prior_visits": {"0": "#C7EA46", "1 - 10": "#4CBB17", "11 - 100": "#0B6623", "100 +": "#043927"},
        "visit_type": {"Inpatient Visit (n=57,552)": "#50C878", "Outpatient Visit (n=1,171,988)": "#7EC0EE", "Emergency Room Visit (n=151,834)": "#D1506F"}
    }
    mapped_ordering = {
        "Gender": ["Male", "Female"],
        "Ethnicity": ["Hispanic", "Not Hispanic"],
        "Race": ["White (n=112,596)", "Asian (n=16,145)", "Black (n=14,154)", "Other (n=25,812)"],
        "Age": ["0 - 17","18 - 34","35 - 64","65 - 99"], #"100 +"],
        "prior_visits": ["0", "1 - 10", "11 - 100", "100 +"],
        "visit_type": ["Outpatient Visit (n=1,171,988)", "Inpatient Visit (n=57,552)", "Emergency Room Visit (n=151,834)"]
    }
    
    sns.set(style="white", font="Utopia", rc={'figure.figsize':(9.3,16)}, font_scale=2)
    sns.boxplot(data=all_bootstrapped_submissions, x="AUROC", y="Team", hue=category_column, fliersize=1, linewidth=0.7, width=0.75, hue_order=mapped_ordering[category_column], palette=dict(colors[category_column]))
    
    
    if category_column == "visit_type":
        plt.legend(title="Last Visit Type", fontsize=15, title_fontsize=20)
    if category_column in ["Ethnicity", "Gender"]:
        plt.xlim(0.7, 1.0)
    elif category_column in ["Age", "visit_type"]:
        pass
    else:
        plt.xlim(0.6, 1.0)
    print (f"{category_column} finished!")

    if not os.path.isdir(f"plots/"):
        os.mkdir(f"plots/")
    plt.savefig(f"plots/{category_column}_bootstrap_{NUM_BOOTSTRAPS}_boxplot.png", bbox_inches="tight")
    plt.clf()
    

def intra_method_distributions(top_subs, predictions, categories, category_column):
    all_bootstrapped_submissions = pd.DataFrame()
    cur_version = get_version(category_column)
    for team in list(top_subs["Team"]):
        team_status = True
        if not os.path.isfile(f"permutations/{team}/{category_column}_{NUM_BOOTSTRAPS}_{cur_version}_bootstraps.csv"):
            print (f"{team} has not been bootstrapped...bootstrapping {team}")
            try:
                bootstrap_submission(team, predictions, categories, category_column, cur_version)
            except KeyError:
                team_status = False
                print (f"{team} does not have a prediction file")
        if team_status:
            scores = pd.read_csv(f"permutations/{team}/{category_column}_{NUM_BOOTSTRAPS}_{cur_version}_bootstraps.csv")
            scores.dropna(inplace=True)
            all_bootstrapped_submissions = pd.concat([all_bootstrapped_submissions, scores])
    print (all_bootstrapped_submissions)
    top_subs["Team"] = top_subs["Team"].astype(str)
    all_bootstrapped_submissions["Team"] = all_bootstrapped_submissions["Team"].astype(str)
    all_bootstrapped_submissions = all_bootstrapped_submissions.merge(top_subs ,left_on="Team", right_on="Team", how="left")
    all_bootstrapped_submissions = all_bootstrapped_submissions.sort_values(f"Score", ascending=False)

    return all_bootstrapped_submissions


def ethnicity_data(data):

    data["ethnicity_concept_id"].fillna(0.0, inplace=True)
    data["ethnicity_concept_id"] = data["ethnicity_concept_id"].astype(int).astype(str)
    data["Ethnicity"] = data["ethnicity_concept_id"].apply(lambda x: cat_descr(x, "ethnicity"))
    data = data[["person_id", "Ethnicity"]]
    
    return data

def gender_submission_data(submissions):
    person  = pd.read_csv(f"{CHALLENGE_DATA}/person.csv")[["person_id", "gender_concept_id"]]
    gold = get_goldstandard()
    data = person.merge(gold, on="person_id", how="right")
    
    preds = collect_predictions_via_team(submissions)
    subs = [i for i in preds.columns if i != "person_id"]

    data = data.merge(preds, on="person_id", how="left")
    #print (list(set(data["gender_concept_id"])))
    data["gender_concept_id"] = data["gender_concept_id"].astype(str)
    data["Gender"] = data["gender_concept_id"].apply(lambda x: cat_descr(x, "gender"))
    gender_cats = get_cats("gender")

    return subs, data, gender_cats, "Gender"

def gender_data(data):

    data["gender_concept_id"] = data["gender_concept_id"].astype(str)
    data["Gender"] = data["gender_concept_id"].apply(lambda x: cat_descr(x, "gender"))
    data = data[["person_id", "Gender"]]

    return data


def age_submission_data(submissions):
    gold = get_goldstandard()
    person  = pd.read_csv(f"{CHALLENGE_DATA}/person.csv")[["person_id", "day_of_birth", "month_of_birth", "year_of_birth"]]
    person_columns = list(person.columns)
    visit   = pd.read_csv(f"{CHALLENGE_DATA}/visit_occurrence.csv")[["person_id","visit_start_date"]]
    data = person.merge(visit, on="person_id", how="left")
    data = data.groupby(person_columns)["visit_start_date"].max()
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)

    data = data.merge(gold, on="person_id", how="right")
    data["visit_start_date"] = pd.to_datetime(data["visit_start_date"])
    data["birth_datetime"] = pd.to_datetime(data.year_of_birth*10000+data.month_of_birth*100+data.day_of_birth,format='%Y%m%d',errors="coerce")
    data["age_at_visit"] = data["visit_start_date"] - data["birth_datetime"]
    data["age_at_visit"] = data["age_at_visit"].apply(lambda x: x.days/365)

    data["Age"] = data["age_at_visit"].apply(lambda x: cat_descr(x, "age"))
    data = data[["person_id", "age_at_visit", "Age", "status"]]
    
    preds = collect_predictions_via_team(submissions)
    subs = [i for i in preds.columns if i != "person_id"]

    data = data.merge(preds, on="person_id", how="left")

    age_cats = get_cats("age")
    
    return subs, data, age_cats, "Age"


def age_data(data):
    
    visit   = pd.read_csv(f"{CHALLENGE_DATA}/visit_occurrence.csv")[["person_id","visit_start_date"]]
    data = data.merge(visit, on="person_id", how="left")
    data = data.groupby(["person_id", "day_of_birth", "month_of_birth", "year_of_birth"])["visit_start_date"].max()
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)

    data["visit_start_date"] = pd.to_datetime(data["visit_start_date"])
    data["birth_datetime"] = pd.to_datetime(data.year_of_birth*10000+data.month_of_birth*100+data.day_of_birth,format='%Y%m%d',errors="coerce")
    data["age_at_visit"] = data["visit_start_date"] - data["birth_datetime"]
    data["age_at_visit"] = data["age_at_visit"].apply(lambda x: round(x.days/365, 0))

    data["Age"] = data["age_at_visit"].apply(lambda x: cat_descr(x, "age"))
    data = data[["person_id", "Age"]]
    
    return data


def race_submission_data(submissions):
    gold = get_goldstandard()
    person  = pd.read_csv(f"{CHALLENGE_DATA}/person.csv")[["person_id", "race_concept_id"]]

    data = person.merge(gold, on="person_id", how="right")
    
    preds = collect_predictions_via_team(submissions)
    subs = [i for i in preds.columns if i != "person_id"]

    data = data.merge(preds, on="person_id", how="left")
    data["race_concept_id"] = data["race_concept_id"].astype(str)
    data["Race"] = data["race_concept_id"].apply(lambda x: cat_descr(x, "Race"))
    
    race_cats = get_cats("race")

    return subs, data, race_cats, "Race"


def race_data(data):
    
    data["race_concept_id"] = data["race_concept_id"].astype(str)
    data["Race"] = data["race_concept_id"].apply(lambda x: cat_descr(x, "Race"))
    data = data[["person_id", "Race"]]

    return data


def visits_prior_to_last_visit():
    visits = pd.read_csv(f"{CHALLENGE_DATA}/visit_occurrence.csv", usecols=["person_id", "visit_occurrence_id", "visit_start_date", "visit_concept_id"], parse_dates=["visit_start_date"])
    
    ## VISIT COUNTS
    visit_counts = pd.DataFrame(visits.groupby("person_id")["visit_occurrence_id"].count())
    visit_counts = visit_counts.reset_index()
    visit_counts["prior_visits"] = visit_counts["visit_occurrence_id"] - 1
    visit_counts["prior_visits_cat"] = visit_counts["prior_visits"].apply(lambda x: cat_descr(x, "prior_visits"))
    visit_counts.drop("visit_occurrence_id", inplace=True, axis=1)

    ## VISIT COUNTS FROM LAST MAX VISIT
    max_visit = pd.DataFrame(visits.groupby("person_id")["visit_start_date"].max().reset_index())
    max_visit.columns = ["person_id", "visit_start_date"]
    max_visit_concepts = max_visit.merge(visits, on=["person_id", "visit_start_date"], how="left")
    max_visit_concepts.drop_duplicates(subset=["person_id", "visit_start_date", "visit_concept_id"], inplace=True)

    max_visit_concepts_ER = max_visit_concepts[max_visit_concepts["visit_concept_id"].isin([9203])]
    max_visit_concepts_IP = max_visit_concepts[max_visit_concepts["visit_concept_id"].isin([9201])]
    max_visit_concepts_IP = max_visit_concepts_IP[~max_visit_concepts_IP["person_id"].isin(set(max_visit_concepts_ER["person_id"]))]
    max_visit_concepts_IP = pd.concat([max_visit_concepts_IP, max_visit_concepts_ER])
    max_visit_concepts_OP = max_visit_concepts[max_visit_concepts["visit_concept_id"].isin([9202])]
    max_visit_concepts_OP = max_visit_concepts_OP[~max_visit_concepts_OP["person_id"].isin(set(max_visit_concepts_IP["person_id"]))]
    max_visit_concepts = pd.concat([max_visit_concepts_IP, max_visit_concepts_OP])

    visit_data = visit_counts.merge(max_visit_concepts, on="person_id", how="outer")
    visit_data["visit_type"] = visit_data["visit_concept_id"].apply(lambda x: cat_descr(x, "visit_type"))

    visit_data = visit_data[["person_id", "prior_visits", "prior_visits_cat", "visit_type"]]

    return visit_data


def gather_patient_information(rebuild=False):
    
    if not os.path.isfile(f"data/annotated_patients.csv") or rebuild:
        print ("Rebuilding patient information")
        eval_pats = pd.read_csv(f"{CHALLENGE_DATA}/person.csv")

        eval_data = eval_pats[["person_id"]]

        visit_eval_data = visits_prior_to_last_visit(eval_pats)
        eval_data = eval_data.merge(visit_eval_data, on="person_id", how="left")
        print ("Visits finished")

        ethnicity_eval_data = ethnicity_data(eval_pats)
        eval_data = eval_data.merge(ethnicity_eval_data, on="person_id", how="left")
        print ("Ethnicity finished")

        race_eval_data = race_data(eval_pats)
        eval_data = eval_data.merge(race_eval_data, on="person_id", how="left")
        print ("Race finished")

        age_eval_data  = age_data(eval_pats)
        eval_data = eval_data.merge(age_eval_data, on="person_id", how="left")
        print ("Age finished")

        gender_eval_data = gender_data(eval_pats)
        eval_data = eval_data.merge(gender_eval_data, on="person_id", how="left")
        print ("Gender finished")

        eval_golds = get_goldstandard()
        eval_data = eval_data.merge(eval_golds, on="person_id", how="left")
        print ("Goldstandard finished")

        print ("Writing patient information file")
        eval_data.to_csv(f"data/annotated_patients.csv", index=False)
    else:
        print ("Reading in built data")
        eval_data = pd.read_csv(f"data/annotated_patients.csv")

    return eval_data


def demographic_boxplot_distributions(cur_categories):
    submissions = get_top_submissions_list().head(10)[["Team", "validation_Score"]]
    submissions.columns = ["Team", "Score"]
    

    patients = gather_patient_information(p, rebuild=False)
    
    predictions = collect_predictions_via_team(submissions)
    
    predictions = predictions.merge(patients, on="person_id", how="left")

    for cat in cur_categories:
        categories = get_cats(cat)
        print (cat, categories)
        
        all_bootstrapped_submissions = intra_method_distributions(submissions, predictions, categories, cat)
        
        graph_boxplots(all_bootstrapped_submissions, cat)


def calibration_curves(subs, data, cats, col):
    #Brier score
    calibrations = pd.DataFrame()
    count = 0
    for sub in subs:
        for c in cats:
            cat_data = data[data[col]==c]
            prob_true, prob_pred = calibration_curve(cat_data["status"], cat_data[f"{sub}"], n_bins=10, strategy="uniform")
            temp = {
                #"Team": [teams[count] for i in range(len(prob_true))],
                "Submission": [sub for i in range(len(prob_true))],
                "Fraction of Positives": prob_true,
                "Mean Predicted Probability": prob_pred,
                col: [c for i in range(len(prob_true))]
            }
            temp = pd.DataFrame(temp)
            calibrations = pd.concat([calibrations, temp])
            count += 1

    calibrations = pd.DataFrame(calibrations)

    top_subs = get_top_submissions_list()

    calibrations = calibrations.merge(top_subs, left_on="Submission", right_on="UW Queue Id", how="left")
    
    color_dict = dict({
        "White":"#E74C3C", 
        "Asian":"#F39C12", 
        "Black":"#2471A3", 
        "Other":"#A6ACAF"
    })

    g = sns.FacetGrid(calibrations, col='Team', hue=col, sharey=True, sharex=True, xlim=(0.0, 1.0), ylim=(0.0, 1.0), col_wrap=5, col_order=list(top_subs["Team"]), aspect=0.9, palette=color_dict)

    g.map(sns.lineplot, x=[0, 1], y=[0, 1], linewidth=0.7, color="#BEBEBE", alpha=0.7)
    g.map(sns.lineplot, "Mean Predicted Probability", "Fraction of Positives", linewidth=2)
    
    g.set_titles("{col_name}", size=15)

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Calibration Curves of top submissions by {col}', fontsize=30)
    
    #plt.legend(ncol=1, fontsize="8", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    g.add_legend(ncol=1, fontsize="15") #, title_fontsize="18")
    plt.setp(g._legend.get_title(), fontsize=18)
    
    plt.savefig(f"plots/calibration_curves_{col}.png", bbox_inches="tight")

