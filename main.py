from src.preprocess import load_and_clean_data
from src.apriori_model import run_apriori
import os

DATA_PATH = "data/real_estate_listings_india_2025.csv"
OUTPUT_PATH = "outputs/association_rules.csv"

def main():

    print("Loading and preprocessing data...")
    df = load_and_clean_data(DATA_PATH)

    print("Running Apriori algorithm...")
    rules = run_apriori(df, min_support=0.02, min_confidence=0.5)

    # Create outputs folder if not exists
    os.makedirs("outputs", exist_ok=True)

    # Save rules
    rules.to_csv(OUTPUT_PATH, index=False)

    print("Top Rules:")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head())

    print(f"\nRules saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()