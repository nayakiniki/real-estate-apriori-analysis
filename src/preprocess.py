import pandas as pd


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop missing values
    df = df.dropna()

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # Convert price into range
    def price_range(price):
        if price < 5000000:
            return "Under_50L"
        elif price < 10000000:
            return "50L_1Cr"
        elif price < 20000000:
            return "1Cr_2Cr"
        else:
            return "Above_2Cr"

    df["PriceRange"] = df["price"].apply(price_range)

    # Convert beds into BHK label
    df["BHK"] = df["beds"].astype(str) + "BHK"

    # Keep only required columns for Apriori
    df = df[["BHK", "PriceRange", "city", "neighborhood", "type"]]

    return df