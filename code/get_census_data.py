# %%
import pandas as pd

# %% England


def remove_wales(df: pd.DataFrame) -> pd.DataFrame:
    """Just a PoC, so we're doing English NHS regions only.

    Args:
        df (pd.DataFrame): data

    Returns:
        pd.DataFrame: data with non-English NHS regions removed
    """
    col = "NHS England regions Code"
    wales = "E40999999"
    df = df.loc[df[col] != wales].reset_index(drop=True)

    return df


def remove_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Data comes with code and category
    e.g. code 4 = Manufacturing. We don't need the code.

    Args:
        df (pd.DataFrame): data

    Returns:
        pd.DataFrame: data with code removed
    """
    code_cols = [i for i in df.columns if i.lower().endswith("code")]
    df = df.drop(code_cols, axis=1)
    return df


def remove_empty_hh(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty households

    Args:
        df (pd.DataFrame): Data

    Returns:
        pd.DataFrame: Data with HH > 0
    """
    col = "Household size (5 categories)"
    empty = "0 people in household"

    if col in df.columns:
        df = df.loc[df[col] != empty].reset_index(drop=True)

    return df


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extraneous info from column headers

    Args:
        df (pd.DataFrame): Data

    Returns:
        pd.DataFrame: Data with cleaned headers
    """
    df.columns = (
        df.columns.str.replace(r"\(.*\)", "", regex=True).str.strip().str.lower()
    )

    df.rename(columns={"observation": "Number of usual residents"})

    return df


def agg_ethnic_group(df: pd.DataFrame) -> pd.DataFrame:
    """Replace granular ethnic groups with smaller subset

    Args:
        df (pd.DataFrame): Data

    Returns:
        pd.DataFrame: Data with aggregated ethnic groups
    """
    rpl = {
        "Asian, Asian British or Asian Welsh": "Asian",
        "Black, Black British, Black Welsh, Caribbean or African": "Black",
        "White: English, Welsh, Scottish, Northern Irish or British": "White British",
        "Other ethnic group: Arab": "Arab",
    }

    for o, n in rpl.items():
        df["ethnic group"] = df["ethnic group"].str.replace(o, n)

    df = df.loc[~df["ethnic group"].str.lower().str.contains("other")]

    df = df.loc[df["ethnic group"] != "Does not apply"].reset_index(drop=True)

    cols = [i for i in df.columns if i != "observation"]

    df = df.groupby(cols, as_index=False)["observation"].sum()
    return df


df = (
    pd.read_csv(r"Census Data/age_sex_ethnic_group_nhs_region.csv")
    .pipe(remove_wales)
    .pipe(remove_codes)
    .pipe(remove_empty_hh)
    .pipe(clean_headers)
    .pipe(agg_ethnic_group)
)


df.to_csv(r"Census Data/cleaned_demographic_data.csv", index=False)
