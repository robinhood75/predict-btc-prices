import requests
from datetime import datetime
import polars as pl
from data.base_loader import DataGenerator


DATA_START_DATE = "2021-01-22"


class DataGeneratorFearAndGreed(DataGenerator):
    def __init__(self, start_date, end_date, data_folder):
        start_date = max(DATA_START_DATE, start_date)
        super().__init__(start_date, end_date, data_folder)

    def load_data(self):
        """
        Récupère les données JSON depuis le site web de CNN.
        Stocke les données brutes (liste de dictionnaires) dans self.raw_data.
        """
        base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        response = requests.get(base_url + self.start_date, headers=headers)
        response.raise_for_status()

        data_json = response.json()
        raw_data = data_json["fear_and_greed_historical"]["data"]

        return raw_data

    def transform_data(self, raw_data):
        """
        Transforme les données brutes en un polars dataFrame 
        """
        # Ajouter les données pour les jours ouvrés
        data = []
        for row in raw_data:
            date = datetime.fromtimestamp(row["x"] / 1000.0)
            date_str = date.strftime('%Y-%m-%d')
            if self.start_date <= date_str <= self.end_date:
                data.append({
                    "date": date_str,
                    "fear_and_greed_score": row["y"],
                    "fear_and_greed_rating": row["rating"]
                })

        ret_data = pl.DataFrame(data)
        ret_schema = pl.DataFrame({"date": pl.Series(self.date_range)})
        ret = ret_schema.join(ret_data, on="date", how="left")

        # Lorsque l'indicateur n'est pas renseigné, renvoyer 50 par défaut
        # et ajouter un flag "fear_and_greed" mis à False
        ret = ret.with_columns(
            pl.when(pl.col("fear_and_greed_score").is_null())
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("fear_and_greed_flag")
        ).with_columns(
            pl.when(pl.col("fear_and_greed_flag").eq(1))
            .then(pl.col("fear_and_greed_score"))
            .otherwise(pl.lit(50.))
            .alias("fear_and_greed_score")
        )

        return ret
