import polars as pl
import os
import alpaca_trade_api as tradeapi
from data.utils import *
from data.base_loader import DataGenerator


class DataGeneratorBTCData(DataGenerator):
    def load_data(self):
        """
        Récupère le cours du BTC depuis l'API Alpaca
        """
        alpaca_api = tradeapi.REST(
            key_id=ALPACA_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_ENDPOINT
        )

        start_iso = f"{self.start_date}T00:00:00Z"
        end_iso = f"{self.end_date}T23:59:59Z"

        bars = alpaca_api.get_crypto_bars(
            symbol="BTC/USD",
            timeframe=tradeapi.TimeFrame.Day,
            start=start_iso,
            end=end_iso
        )

        return bars

    def transform_data(self, raw_data):
        """
        Colonnes du dataframe polars retourné : 
        "date", "open", "high", "low", "close", "volume"
        """
        data = []
        for bar in raw_data:
            data.append({
                "date":   bar.t.date().isoformat(),
                "open":   bar.o,
                "high":   bar.h,
                "low":    bar.l,
                "close":  bar.c,
                "volume": bar.v,
            })

        df = pl.DataFrame(data)
        df = self._add_log_returns(df)
        df = df.rename({col: f"btc_{col}" for col in df.columns if col != "date"})

        return df
    
    @staticmethod
    def _add_log_returns(df, cols=["open", "close"]):
        """
        Pour chaque colonne de @param cols, ajoute à @param df une colonne
        f"r_{col}" calculant ln(col[t]/col[t-1])
        Return : df avec les colonnes supplémentaires
        """
        for col in cols:
            df = df.with_columns(
                (pl.col(col) / pl.col(col).shift(1))
                .log()
                .alias(f"r_{col}")
            )
        return df
