import numpy as np
import pandas as pd

def shuffle(df, n):
    np.random.seed(1313)
    remove_n = n
    random_indices = np.random.choice(df.index, remove_n, replace=False)

    df1 = df.drop(random_indices)
    df2 = df.iloc[random_indices, :]

    df2.columns = df2.columns.str.replace("team1","team3",regex=True)
    df2.columns = df2.columns.str.replace("team2","team1",regex=True)
    df2.columns = df2.columns.str.replace("team3","team2",regex=True)

    df = pd.concat([df2, df1]).reset_index(drop=True)

    return df