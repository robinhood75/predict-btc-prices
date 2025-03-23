import requests
import polars as pl
import os
from datetime import datetime
from data.utils import get_dates_range
from data.base_loader import DataGenerator


ENDPOINTS = [
    "realized-cap-hodl-waves",
    "aviv",
    "hashrate",
    "hashprice",
    "mvocdd",
    "mvrv",
    "mvrv-zscore",
    "nvt-ratio",
    "cap-real-usd",
    "vocdd",
    "cdd-90dma",
    "true-market-mean",
    "thermo-cap",
    "reserve-risk",
    "nupl",
    "miner-reserves",
    "out-flows",
    "coins-addr-1-BTC",
    "coins-addr-10-1-BTC",
    "coins-addr-100-10-BTC",
    "coins-addr-10K-1K-BTC",
    "coins-addr-1K-100-BTC",
    "balance-addr-1-BTC",
    "balance-addr-10-1-BTC",
    "balance-addr-100-10-BTC",
    "balance-addr-10K-1K-BTC",
    "balance-addr-1K-100-BTC",
]


class DataGeneratorOnChainData(DataGenerator):
    """
    Récupère de nombreuses données on-chain sur le Bitcoin depuis
    l'API bitcoin-data.com
    """
    def __init__(self, start_date, end_date, data_folder, endpoints):
        super().__init__(start_date, end_date, data_folder)
        self.endpoints = endpoints

    def load_data(self):
        base_url = "https://bitcoin-data.com"

        raw_data = [self._load_data(base_url, endpoint) for endpoint in self.endpoints]

        return raw_data
    
    def _load_data(self, base_url, endpoint):
        """Récupère les données pour un endpoint spécifique"""
        url = f"{base_url}/v1/{endpoint}"
        params = {
            "startday": self.start_date,
            "endday": self.end_date
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"Erreur API : {response.status_code}, {response.text}")

        return response.json()

    def transform_data(self, raw_data):
        combined_data = []

        for endpoint_data in raw_data:
            df = pl.DataFrame(endpoint_data).drop("unixTs")
            col_day = "d" if "d" in df.columns else "theDay"
            df = df.rename({col_day: "date"})
            df = df.with_columns(
                [pl.col(col).cast(pl.Float64) for col in df.columns if col not in ["date"]]
            )
            combined_data.append(df)
        
        ret = combined_data[0]
        if len(combined_data) > 1:
            for df in combined_data[1:]:
                ret = ret.join(df, how="left", on="date")
        
        return ret
