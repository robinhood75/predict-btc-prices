import os
import datetime as dt
import polars as pl
from data.utils import get_dates_range, vectorize_dates


def make_dataset(sources, data_folder):
    """
    Création du dataframe final à partir des différentes sources de données
    @param sources: liste de sources de données au format str (exemple : "news")
    @param data_folder: dossier dans lequel les données sont localisées
    """
    # Structurer les données disponibles par catégorie
    data_files = [f for f in os.listdir(data_folder) if f.endswith(".parquet")]
    data_list = [
        pl.concat(
            [pl.read_parquet(os.path.join(data_folder, f)) 
             for f in data_files if source_name in f],
            how="vertical_relaxed"
        )
        for source_name in sources
    ]

    start_date = min([df["date"].min() for df in data_list])
    end_date = max([df["date"].max() for df in data_list])
    dates_list = get_dates_range(start_date, end_date)
    ret = pl.DataFrame({"date": dates_list})

    # Jointure des sources de données
    for df_source in data_list:
        ret = ret.join(df_source, on="date")

    # Dupliquer toutes les colonnes, shiftées à j-1
    ret = ret.with_columns(
        [pl.col(col).shift(1).alias(f"{col} hier") for col in ret.columns if col != "date"]
    )

    # One-hot-encoding de la date
    ret = vectorize_dates(ret)

    return ret
