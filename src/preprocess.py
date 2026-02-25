import pandas as pd


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    # Convert price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # Price segmentation
    def price_segment(price):
        if price < 5000000:
            return "Budget"
        elif price < 10000000:
            return "MidRange"
        elif price < 20000000:
            return "Premium"
        else:
            return "Luxury"

    df["PriceSegment"] = df["price"].apply(price_segment)

    # BHK label
    df["BHK"] = df["beds"].astype(str) + "BHK"

    # Size categorization
    df["size_sqft"] = df["size"].str.extract(r"(\d+)").astype(float)

    def size_category(size):
        if size < 800:
            return "Small"
        elif size < 1500:
            return "Medium"
        else:
            return "Large"

    df["SizeCategory"] = df["size_sqft"].apply(size_category)

    # Property category simplified
    df["PropertyCategory"] = df["type"].str.replace(" ", "_")

    df = df[[
        "BHK",
        "PriceSegment",
        "city",
        "neighborhood",
        "SizeCategory",
        "PropertyCategory"
    ]]

    return df