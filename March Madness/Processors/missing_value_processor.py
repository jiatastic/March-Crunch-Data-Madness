import pandas as pd

redundancy = ["team1_id","team2_id","game_id","team2_teamname","team1_teamname","slot","team1_score","team2_score","team2_coach_id","team1_coach_id"]

def ratio(df):
    df.drop(columns = redundancy, inplace = True)

    a = df.isnull().sum()/len(df)*100

    variables = df.columns
    variable = []

    for i in range(len(variables)):
        if a[i]<=20:   #setting the threshold as 20%
            variable.append(variables[i])

    df = df[variable]

    return df