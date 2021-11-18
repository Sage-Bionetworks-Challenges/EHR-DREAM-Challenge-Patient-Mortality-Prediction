from functools import reduce
import pandas as pd
import json


## Gather the predictions from the run docker models.
## Outputs a two column dataframe with person_id and prediction scores.
def collect_predictions_via_team(top):
    predList = []
    teams = list(top["Team"])

    model_output = json.load(open("config.json"))["MODEL_OUTPUT"]

    count = 0
    for team in teams:
        
        pred = pd.read_csv(f"{model_output}/{team}/predictions.csv")
        
        predList.append(pred)
        pred.columns = ["person_id", f"{team}"]
        count += 1
    preds = reduce(lambda x, y: pd.merge(x, y, on = 'person_id'), predList)
    preds = preds.astype(float)
    preds["person_id"] = preds["person_id"].astype(int)

    return preds


## Gather the true answers for the patient outcomes
def get_goldstandard():
    goldstandard_path = json.load(open("config.json"))["GOLDSTANDARD"]
    gold = pd.read_csv(goldstandard_path)
    
    return gold


## gather the model names and information
def get_top_submissions_list():
    submissions = pd.read_csv("data/top_submissions.csv")
    submissions["UW Queue Id"] = submissions["UW Queue Id"].astype(str)
    return submissions