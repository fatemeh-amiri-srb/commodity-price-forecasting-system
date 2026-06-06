from src.data_loader import load_data
from src.train import train_model
from src.predict import forecast

if __name__ == "__main__":

    df = load_data("data/raw/soybean_data.xlsx")

    model, features, processed_df = train_model(df)

    forecast(model, features, processed_df)