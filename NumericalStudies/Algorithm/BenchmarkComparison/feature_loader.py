def _feature_index(column_name):
    return int(column_name.split("_", 1)[1])


def load_feature_matrix(df, config):
    available_feature_cols = sorted(
        [col for col in df.columns if col.startswith("Feature_")],
        key=_feature_index,
    )

    feature_cols = available_feature_cols

    return df[feature_cols].values.astype(float), feature_cols
