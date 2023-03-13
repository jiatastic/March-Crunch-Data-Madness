from math import sin, cos, sqrt, atan2, radians


def distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance




def feature_engineering(df):
    df["sead_diff"] = df["team1_seed"] - df["team2_seed"]

    df['team1_region'] = df["team1_position"].str[:1]
    df['team2_region'] = df["team2_position"].str[:1]

    df.drop(columns=['team1_position', 'team2_position'], inplace=True)

    # Expected Win Rates
    df['exp_win1'] = (df['team1_adjoe'] ** 11.5) / ((df['team1_adjde'] ** 11.5) + (df['team1_adjoe'] ** 11.5))
    df['exp_win2'] = (df['team2_adjoe'] ** 11.5) / ((df['team2_adjde'] ** 11.5) + (df['team2_adjoe'] ** 11.5))

    df['dist1'] = df.apply(lambda row: distance(row['host_lat'], row['host_long'], row['team1_lat'], row['team1_long']),
                           axis=1)
    df['dist2'] = df.apply(lambda row: distance(row['host_lat'], row['host_long'], row['team2_lat'], row['team2_long']),
                           axis=1)

    df['diff_dist'] = df['dist1'] - df['dist2']

    df['team1_win'] = (df['team1_score'] > df['team2_score']).astype(int)

    return df
