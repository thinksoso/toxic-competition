def lower_case(df):
    df["more_toxic"] = df["more_toxic"].str.lower()
    df["less_toxic"] = df["less_toxic"].str.lower()
    return df


def special_filter():
    pass


def lowercase():
    pass


def stop_word():
    pass


def preprocess(df):
    df = lower_case(df)
    return df
