from datetime import datetime, timedelta
from polars import DataFrame
import os
from data.utils import get_dates_range


class DataGenerator:
    def __init__(self, start_date, end_date, data_folder):
        """
        Load data from API from start_date (included) to end_date (included),
        and store it in data_folder
        """
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = get_dates_range(self.start_date, self.end_date)
        self.data_folder = data_folder

    def load_data(self):
        raise NotImplementedError
    
    def transform_data(self, raw_data):
        raise NotImplementedError
    
    def save_df(self, data, name):
        parquet_path = os.path.join(self.data_folder, f"{name}_{self.start_date}_{self.end_date}.parquet")
        if isinstance(data, DataFrame):
            data.write_parquet(parquet_path)
