import holidays
import json
import os
import polars as pl
from datetime import datetime, timedelta


# API keys
keys_file_path = os.path.join("data", "keys.json")
with open(keys_file_path, "r") as f:
    dict_keys = json.load(f)

NEWSAPI_KEY = dict_keys.get("news_api")
OPENAI_KEY = dict_keys.get("openai_api")

alpaca_keys = dict_keys.get("alpaca_trading_api")
ALPACA_ENDPOINT = alpaca_keys.get("endpoint")
ALPACA_KEY = alpaca_keys.get("key")
ALPACA_SECRET_KEY = alpaca_keys.get("secret_key")

LUNARCRUSH_KEY = dict_keys.get("lunarcrush_api")


# News loading parameterization
news_categories_path = os.path.join("data", "categories_news.json")
with open(news_categories_path, "r") as f:
    NEWS_PARAMETERS = json.load(f)


# Sentiment map
SENTIMENT_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}


def get_dates_range(start_date, end_date):
    """
    Return list of dates between start_date and end_date with format 'yyyy-mm-dd'
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = []
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return date_list


def vectorize_dates(df, col_date="date"):
    """
    @param df: dataframe Polars avec une colonne @param col_date au format str
    yyyy-mm-dd
    Retourne le même dataframe avec un one-hot-encoding du jour de la semaine,
    du numéro de la semaine dans le mois, et du mois
    """
    df = df.with_columns(pl.col(col_date).str.to_datetime("%Y-%m-%d").alias(col_date))

    # Ajout colonnes jour, semaine, mois
    df = df.with_columns([
        pl.col(col_date).dt.weekday().alias("jour"),
        (
            (pl.col(col_date).dt.day() + pl.col(col_date).dt.truncate("1mo").dt.weekday() - 2) // 7 + 1
        ).alias("semaine"),
        pl.col(col_date).dt.month().alias("mois")
    ])
    
    # One-hot-encoding de jour, semaine mois
    jour_map = {1: "lundi", 2: "mardi", 3: "mercredi", 4: "jeudi", 5: "vendredi", 6: "samedi", 7: "dimanche"}
    mois_map = {
        1: "janvier", 2: "fevrier", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
        7: "juillet", 8: "aout", 9: "septembre", 10: "octobre", 11: "novembre", 12: "decembre"
    }
    df = df.with_columns([
        pl.col("jour").map_elements(lambda x: jour_map.get(x, "inconnu"), return_dtype=pl.Utf8),
        pl.col("mois").map_elements(lambda x: mois_map.get(x, "inconnu"), return_dtype=pl.Utf8)
    ])
    df = df.with_columns([
        pl.col("jour").cast(pl.Categorical),
        pl.col("mois").cast(pl.Categorical)
    ])

    df = df.to_dummies("jour")
    df = df.to_dummies("semaine")
    df = df.to_dummies("mois")
    
    # Ajout jours fériés aux Etats-Unis
    us_holidays = holidays.US()
    df = df.with_columns(
        pl.col(col_date)
        .map_elements(lambda d: 1 if d in us_holidays else 0, return_dtype=pl.Int64)
        .alias("jour_ferie_us")
    )
    
    # Typage de la date
    df = df.with_columns(pl.col(col_date).dt.strftime("%Y-%m-%d"))

    return df
