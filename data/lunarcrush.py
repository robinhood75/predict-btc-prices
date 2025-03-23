import requests
import os
from datetime import datetime
import polars as pl

from data.utils import get_dates_range
from data.base_loader import DataGenerator
from data.utils import *


class DataGeneratorLunarCrush(DataGenerator):
    """
    Récupère les données journalières de LunarCrush pour un symbole donné
    sur la plage de dates [start_date, end_date], puis enregistre le tout
    au format Parquet dans data_folder.
    """
    def __init__(self, start_date, end_date, data_folder, hourly=False):
        self.hourly = hourly
        super().__init__(start_date, end_date, data_folder)

    def load_data(self):
        """
        Récupère les données journalières du Bitcoin au format JSON
        """
        authentification = {
            "Authorization": f"Bearer {LUNARCRUSH_KEY}"
        }

        bucket_name = "hour" if self.hourly else "day"
        url = f"https://lunarcrush.com/api4/public/coins/btc/time-series/v2"
        params = {
            "bucket": bucket_name,
            "start": int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp()),
            "end": int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp())
        }

        response = requests.get(url, headers=authentification, params=params)
        if response.status_code != 200:
            raise Exception(
                f"Erreur lors de l'appel à l'API LunarCrush. "
                f"Code HTTP: {response.status_code}, Réponse: {response.text}"
            )

        raw_data = response.json()
        raw_data = raw_data.get("data")

        return raw_data


    def transform_data(self, raw_data):
        """
        Transforme la réponse brute (JSON) de LunarCrush en un DataFrame polars.
        """
        df = pl.DataFrame(raw_data)

        df = df.with_columns(
            pl.col("time").apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")).alias("date")
        )
        df = add_log_returns(df)

        return df
    