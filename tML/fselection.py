from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

def select_features_by_corr(df_all, A_set, B_dummy_cols, top_n_A, top_n_B, target_col='badflag', corr_threshold=0.0):
    """
    Select top correlated features from A_set and B_dummy_cols using absolute Pearson correlation.
    """
    def compute_corr(features):
        corr_list = []
        for f in features:
            if df_all[f].isnull().all():
                continue
            corr = abs(df_all[f].corr(df_all[target_col]))
            if corr >= corr_threshold:
                corr_list.append((f, corr))
        return sorted(corr_list, key=lambda x: x[1], reverse=True)

    A_top = [f for f, _ in compute_corr(A_set)[:top_n_A]]
    B_top = [f for f, _ in compute_corr(B_dummy_cols)[:top_n_B]]
    return A_top, B_top, A_top + B_top

def select_features_by_mi(df_all, A_set, B_dummy_cols, top_n_A, top_n_B, target_col='badflag'):
    """
    Select top features using mutual information with the target.
    """
    def compute_mi(features):
        X = df_all[features].fillna(0)
        y = df_all[target_col]
        mi = mutual_info_classif(X, y, discrete_features='auto')
        return sorted(zip(features, mi), key=lambda x: x[1], reverse=True)

    A_top = [f for f, _ in compute_mi(A_set)[:top_n_A]]
    B_top = [f for f, _ in compute_mi(B_dummy_cols)[:top_n_B]]
    return A_top, B_top, A_top + B_top

def select_features_by_anova(df_all, A_set, B_dummy_cols, top_n_A, top_n_B, target_col='badflag'):
    """
    Select top features using ANOVA F-value between feature and target.
    """
    def compute_anova(features):
        X = df_all[features].fillna(0)
        y = df_all[target_col]
        f_values, _ = f_classif(X, y)
        return sorted(zip(features, f_values), key=lambda x: x[1], reverse=True)

    A_top = [f for f, _ in compute_anova(A_set)[:top_n_A]]
    B_top = [f for f, _ in compute_anova(B_dummy_cols)[:top_n_B]]
    return A_top, B_top, A_top + B_top
