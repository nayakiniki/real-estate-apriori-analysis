import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(df, min_support=0.02, min_confidence=0.5):

    # Create transaction list
    transactions = df.values.tolist()

    # One-hot encoding
    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)

    # Apply Apriori
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    # Generate rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Sort by lift
    rules = rules.sort_values(by="lift", ascending=False)

    return rules