from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import NON_PRODUCT_ENTRIES, ProjectPaths


@dataclass
class DatasetBundle:
    total_retail: pd.DataFrame
    all_products: pd.DataFrame
    retail_train: pd.DataFrame
    retail_test: pd.DataFrame
    all_weeks: pd.PeriodIndex
    training_weeks: pd.PeriodIndex
    test_weeks: pd.PeriodIndex


def _load_cleaned_csvs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_retail = pd.read_csv(paths.cleaned_retail_csv, parse_dates=["InvoiceDate"])
    all_products = pd.read_csv(paths.cleaned_products_csv)
    total_retail["StockCode"] = total_retail["StockCode"].astype(str)
    total_retail["Invoice"] = total_retail["Invoice"].astype(str)
    total_retail["Description"] = total_retail["Description"].astype(str)
    total_retail["CustomerID"] = total_retail["CustomerID"].astype(str)
    total_retail["Country"] = total_retail["Country"].astype(str)
    return total_retail, all_products


def _clean_raw_excel(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not paths.raw_excel.exists():
        raise FileNotFoundError(
            "Raw Excel file not found. Place 'online_retail_II.xlsx' in 'data/raw/', "
            "set FORTUNETELLERS_RAW_EXCEL, or pass --raw-excel explicitly."
        )
    xl_file = pd.ExcelFile(paths.raw_excel)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}

    year_2009 = dfs["Year 2009-2010"].copy()
    year_2009["Invoice"] = year_2009["Invoice"].astype(str)
    first_repeat_idx = year_2009[year_2009["Invoice"] == "536365"].index[0]
    year_2009 = year_2009.iloc[:first_repeat_idx]

    total_retail = pd.concat([year_2009, dfs["Year 2010-2011"]], ignore_index=True)
    total_retail["InvoiceDate"] = pd.to_datetime(total_retail["InvoiceDate"])
    total_retail[["Invoice", "StockCode", "Description"]] = total_retail[
        ["Invoice", "StockCode", "Description"]
    ].astype(str)
    total_retail.columns = [col.replace(" ", "").replace("_", "") for col in total_retail.columns]
    total_retail[["CustomerID", "Country"]] = total_retail[["CustomerID", "Country"]].astype(str)

    total_retail = total_retail.drop_duplicates()
    total_retail = total_retail[total_retail["Price"] > 0].copy()

    all_products = (
        total_retail.groupby("StockCode")["Description"]
        .agg(["first", "count"])
        .sort_values("count", ascending=False)
        .reset_index()
    )
    all_products = all_products[~all_products["first"].isin(NON_PRODUCT_ENTRIES)].copy()
    total_retail = total_retail[total_retail["StockCode"].isin(all_products["StockCode"])].copy()

    total_retail["CustomerID"] = total_retail["CustomerID"].astype(str)
    total_retail["Country"] = total_retail["Country"].astype(str)

    paths.ensure_dirs()
    total_retail.to_csv(paths.cleaned_retail_csv, index=False)
    all_products.to_csv(paths.cleaned_products_csv, index=False)
    return total_retail, all_products


def _add_time_columns(total_retail: pd.DataFrame) -> pd.DataFrame:
    df = total_retail.copy()
    df["Week"] = df["InvoiceDate"].dt.to_period("W")
    df["Sales"] = df["Quantity"] * df["Price"]
    return df


def load_or_prepare_transactions(paths: ProjectPaths, forecast_horizon: int = 12) -> DatasetBundle:
    paths.ensure_dirs()
    if paths.cleaned_retail_csv.exists() and paths.cleaned_products_csv.exists():
        total_retail, all_products = _load_cleaned_csvs(paths)
    else:
        total_retail, all_products = _clean_raw_excel(paths)

    total_retail = _add_time_columns(total_retail)
    all_weeks = pd.period_range(
        start=total_retail["Week"].min(),
        end=total_retail["Week"].max(),
        freq="W",
    )
    training_weeks = all_weeks[:-forecast_horizon]
    test_weeks = all_weeks[-forecast_horizon:]

    retail_train = total_retail[total_retail["Week"].isin(training_weeks)].copy().reset_index(drop=True)
    retail_test = total_retail[total_retail["Week"].isin(test_weeks)].copy().reset_index(drop=True)

    return DatasetBundle(
        total_retail=total_retail,
        all_products=all_products,
        retail_train=retail_train,
        retail_test=retail_test,
        all_weeks=all_weeks,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
    )
