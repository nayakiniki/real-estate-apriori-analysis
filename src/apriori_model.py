import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(df, min_support=0.03, min_confidence=0.6):

    transactions = df.values.tolist()

    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)

    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Remove weak rules
    rules = rules[rules["lift"] > 1.2]

    rules = rules.sort_values(by="lift", ascending=False)

    return rules