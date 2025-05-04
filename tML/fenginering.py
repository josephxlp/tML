import pandas as pd
import numpy as np
from itertools import combinations

# --- Feature engineering: transformations ---
def feature_engineering(df, variables, operations=["subtract"], winsorize=True):
    df = df.copy()
    new_features = {}

    for var in variables:
        if var not in df.columns:
            continue
        series = df[var]
        mean_val = series.mean(skipna=True)

        for op in operations:
            if op == "subtract":
                new_features[f"{var}_minus_mean"] = series - mean_val
            elif op == "divide" and mean_val != 0:
                new_features[f"{var}_divided_by_mean"] = series / mean_val
            elif op == "log":
                mask = series > 0
                log_values = pd.Series(np.nan, index=df.index)
                log_values[mask] = np.log(series[mask])
                new_features[f"log_{var}"] = log_values
            elif op == "square":
                new_features[f"{var}_squared"] = series ** 2
            else:
                print(f"Unsupported operation '{op}' for variable '{var}'")

        # Winsorization: clip to the 1st and 99th percentiles
        if winsorize:
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            winsorized = series.clip(lower, upper)
            new_features[f"{var}_winsorized"] = winsorized

    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

# --- Feature engineering: pairwise combinations ---
def arithmetic_operations(df, variables, operations=["divide"]):
    df = df.copy()
    new_features = {}
    for var1, var2 in combinations(variables, 2):
        if var1 not in df.columns or var2 not in df.columns:
            continue
        for op in operations:
            if op == "multiply":
                new_features[f"{var1}_times_{var2}"] = df[var1] * df[var2]
            elif op == "subtract":
                new_features[f"{var1}_minus_{var2}"] = df[var1] - df[var2]
            elif op == "divide":
                new_features[f"{var1}_divided_by_{var2}"] = np.where((df[var2] == 0) | (df[var2].isna()), np.nan, df[var1] / df[var2])
            elif op == "add":
                new_features[f"{var1}_plus_{var2}"] = df[var1] + df[var2]
            else:
                print(f"Unsupported operation '{op}'")
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

"""sumary_line
base_vars = train_df.drop([tcol,idcol],axis=1).columns.tolist()
train_df = feature_engineering(train_df, base_vars, operations=["subtract", "divide", "log", "square"])
train_df = arithmetic_operations(train_df, base_vars, operations=["multiply", "subtract", "divide", "add"])
"""
