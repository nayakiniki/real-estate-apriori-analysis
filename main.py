from src.preprocess import load_and_clean_data
from src.apriori_model import run_apriori
import os


DATA_PATH = "data/real_estate_listings_india_2025.csv"
OUTPUT_PATH = "outputs/association_rules.csv"


def print_top_insights(rules, top_n=5):
    print("\n==============================")
    print("Top Strong Market Patterns")
    print("==============================\n")

    for index, row in rules.head(top_n).iterrows():
        antecedents = ", ".join(list(row["antecedents"]))
        consequents = ", ".join(list(row["consequents"]))

        print(f"Rule: {antecedents}  →  {consequents}")
        print(f"Support: {row['support']:.3f}")
        print(f"Confidence: {row['confidence']:.3f}")
        print(f"Lift: {row['lift']:.3f}")
        print("-----------------------------------")


def city_wise_analysis(df):
    print("\n==============================")
    print("City-Wise Market Distribution")
    print("==============================\n")

    city_counts = df["city"].value_counts()

    for city, count in city_counts.items():
        print(f"{city}: {count} listings")


def segment_distribution(df):
    print("\n==============================")
    print("Price Segment Distribution")
    print("==============================\n")

    segment_counts = df["PriceSegment"].value_counts()

    for segment, count in segment_counts.items():
        print(f"{segment}: {count} properties")


def main():

    print("Loading and preprocessing data...\n")
    df = load_and_clean_data(DATA_PATH)

    print("Running Advanced Apriori Analysis...\n")
    rules = run_apriori(
        df,
        min_support=0.03,
        min_confidence=0.6
    )

    if rules.empty:
        print("No strong rules found. Try lowering support/confidence.")
        return

    # Create output folder if not exists
    os.makedirs("outputs", exist_ok=True)

    # Save rules
    rules.to_csv(OUTPUT_PATH, index=False)

    print_top_insights(rules, top_n=5)

    city_wise_analysis(df)

    segment_distribution(df)

    print("\nAnalysis completed successfully.")
    print(f"Association rules saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()