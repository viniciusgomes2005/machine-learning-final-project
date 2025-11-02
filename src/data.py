import pandas as pd
from dateutil.parser import parse

def load_data(path):
    df = pd.read_csv(path)
    return df

def parse_time(series):
    return series.apply(lambda x: parse(str(x)) if pd.notna(x) else pd.NaT)
