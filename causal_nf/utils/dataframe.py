import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import matplotlib.pyplot as plt
from matplotlib import colors

import causal_nf.utils.io as causal_io


def parameters_cross_validated(df):
    params = {}
    columns = sorted(list(df.columns))
    colums__ = [c for c in columns if "__" in c]
    for c in colums__:
        if c in ["dataset__k_fold"]:
            continue

        try:
            unique_values = df[c].unique()
        except:
            causal_io.print_warning(f"Could not get unique of column {c}")
            continue
        if len(unique_values) > 1:
            params[c] = unique_values

    return params


def show_filter_options(df: pd.DataFrame, to_filter_columns):
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    for column in to_filter_columns:
        print(f"{column} [{df[column].dtype}]")
        values_str = None

        # Treat columns with < 10 unique values as categorical
        if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
            values_str = ""
            for i in df[column].unique():
                values_str += f"\t {i}\n"
        elif is_numeric_dtype(df[column]):
            _min = float(df[column].min())
            _max = float(df[column].max())
            values_str = f"\t {_min} - {_max}"

        elif is_datetime64_any_dtype(df[column]):

            start_date = df[column].min()
            end_date = df[column].max()
            values_str = f"\t {start_date} - {end_date}"
        else:
            try:
                values_str = ""
                for i in df[column].unique():
                    values_str += f"\t {i}\n"
            except Exception:
                pass

        if values_str:
            print(values_str)


def filter_df(df, filter_):
    for c, values in filter_.items():
        df = df[df[c].isin(values)]
    return df


def filter_dataframe(df: pd.DataFrame, columns_filter) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    for column, values in columns_filter.items():
        # Treat columns with < 10 unique values as categorical
        if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
            df = df[df[column].isin(values)]
        elif is_numeric_dtype(df[column]):
            df = df[df[column].between(*values)]
        elif is_datetime64_any_dtype(df[column]):

            if len(values) == 2:
                user_date_input = tuple(map(pd.to_datetime, values))
                start_date, end_date = user_date_input
                df = df.loc[df[column].between(start_date, end_date)]
        else:
            df = df[df[column].astype(str).str.contains(values)]

    return df


def style_df(df, df_ref=None, cmap="RdYlGn"):
    if df_ref is None:
        return df.style.background_gradient(axis=0, cmap=cmap)

    def b_g(s, cmap, low=0, high=0):
        # Pass the columns from Dataframe A
        col = s.name
        a = df_ref.loc[:, s.name].copy()
        rng = a.max() - a.min()
        norm = colors.Normalize(a.min() - (rng * low), a.max() + (rng * high))
        normed = norm(a.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ["background-color: %s" % color for color in c]

    return df.style.apply(b_g, cmap=cmap)


def create_mean_std_df(df, groupby_cols, metric_cols, precision=2, aggfunc="mean"):
    cols = [*groupby_cols, *metric_cols]
    df_mean = getattr(df[cols].groupby(groupby_cols)[metric_cols], aggfunc)()
    df_std = df[cols].groupby(groupby_cols)[metric_cols].std()
    # create a new empty dataframe with the same shape as df_1 and df_2
    new_df = pd.DataFrame(columns=df_std.columns, index=df_mean.index)

    # iterate over the rows and columns of the dataframes
    for idx in df_mean.index:
        for col in df_mean.columns:
            # concatenate the corresponding cells with a '+-' separator
            mu_ = round(df_mean.loc[idx, col], precision)
            std_ = round(df_std.loc[idx, col], precision)
            new_cell = f"{mu_} $\pm$ {std_}"
            # set the value of the corresponding cell in the new dataframe
            new_df.at[idx, col] = new_cell

    return style_df(df=new_df, df_ref=df_mean)


def convert_nodes_to_df(G, node_id_col="node_id"):
    data = []

    for node_id, attr_dict in G.nodes(True):
        my_dict = {node_id_col: node_id}
        my_dict.update(attr_dict)
        my_dict["in_degree"] = G.in_degree(node_id)
        my_dict["out_degree"] = G.out_degree(node_id)
        data.append(my_dict)
    return pd.DataFrame(data)


def convert_edges_to_df(G):
    data = []
    for node_src, node_dst in G.edges:
        my_dict = {"node_src": node_src, "node_dst": node_dst}
        my_dict.update(G.edges[(node_src, node_dst)])
        data.append(my_dict)
    return pd.DataFrame(data)
