# predict-btc-prices
This repository attempts to predict daily BTC prices by considering the following data : sentiment analysis of news from relevant topics, BTC on-chain data, LunarCrush's social indices, CNN's Fear & Greed index. This is how it should be used:
- **dataset.ipynb** allows you to consolidate a dataset stored in data/parquet/dataset.parquet with dozens of daily predictors. Fill the data/keys.json file to access the necessary APIs to generate this data.
- **model testing.ipynb** is used by instantiating a ML model (declared in src/models.py), and specifying training settings in src/config.json. Visualization tools for training monitoring are provided.

Feel free to contact me if you identify any potential improvement on this approach.
