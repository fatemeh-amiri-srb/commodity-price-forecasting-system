import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load and clean commodity dataset
    """

    df = pd.read_excel(path)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    # numeric cleaning
    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.dropna().copy()

    return df