import json
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def read_config(json_path):
    """
    Lecture du fichier JSON contenant les infos de config
    Retourne un dictionnaire
    """
    with open(json_path, 'r') as f:
        ret = json.load(f)
        
    params_obligatoires = [
        "data_file",
        "features",
        "target"
    ]
    params_optionnels_defaults = {
        "k": 8,
        "metric": "R2",
        "normalize_features": True,
        "normalize_target": False,
        "nulls_handling": "drop",
        "rnn_seq_length": 5
    }
    
    # S'assurer que les paramètres obligatoires sont présents
    for param in params_obligatoires:
        if param not in ret.keys():
            raise ValueError(f"Param {param} non renseigné dans config.json")
            
    # Remplacer les paramètres optionnels non renseignés par leurs defaults
    for param in params_optionnels_defaults.keys():
        if param not in ret.keys():
            ret[param] = params_optionnels_defaults[param]
        
    return ret


def read_decoupage(decoupage_path, X_columns):
    """
    Lecture du fichier JSON contenant les infos de découpage des features par sous-modèle
    Retourne un dictionnaire
    """
    with open(decoupage_path, 'r') as f:
        ret = json.load(f)
    mapping = {col_name: i for i, col_name in enumerate(X_columns)}
    for source, cols in ret.items():
        ret[source] = [mapping[col_name] for col_name in cols]
    return ret


def compute_score(y_pred, y_test, target_name, metric):
    """
    Calcul du score pour la métrique choisie ("RMSE", "R2", "MAE" ou "accuracy")
    /!\ Calcul sur les jours ouvrés uniquement
    """
    y = y_test.with_columns(
        pl.Series(y_pred)
        .alias("prediction")
    )
    
    # Garder uniquement les jours ouvrés
    truth = y[target_name].to_numpy()
    pred = y["prediction"].to_numpy()
    
    # Calcul du score
    if metric == "RMSE":
        score = np.sqrt(mean_squared_error(pred, truth))
    elif metric == "R2":
        score = r2_score(truth, pred)
    elif metric == "MAE":
        score = mean_absolute_error(pred, truth)
    elif metric == "accuracy":
        pred = (np.sign(pred) + 1) / 2
        score = np.mean(truth == pred)
    else:
        raise ValueError(f"Metric choisie '{metric}' inconnue")
        
    return score


def get_normalized_data(df):
    """
    Mettre la moyenne du df target à 0, et sa variance à 1
    """
    print(df)
    df_std = df.with_columns(
        [((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
            .alias(col)
            for col in df.columns]
    )
    
    mean = df.mean()
    std = df.std()
    
    return df_std, mean, std


def get_rnn_input(X, rnn_seq_length):
    """
    Chaque élément devient une liste des rnn_seq_length derniers éléments
    """
    # Chaque élément devient une liste de longueur rnn_seq_length
    if rnn_seq_length is not None:
        X = X.with_columns([
            pl.concat_list(
                np.flip([pl.col(c).shift(i) for i in range(rnn_seq_length)])
            ).alias(c)
            for c in X.columns
        ])
        X = X.tail(-rnn_seq_length) # supprimer les 1ères lignes (nulls)
        y = y.tail(-rnn_seq_length)