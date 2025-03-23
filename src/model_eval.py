import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from copy import copy, deepcopy
from src.log import logger
from src.utils import read_config, compute_score
from sklearn.model_selection import train_test_split, learning_curve


def get_model_eval(model, config_path, seed):
    """
    Retourne une instance de ModelEval à partir du fichier config et 
    d'une seed
    """
    # Lecture du fichier config
    params = read_config(config_path)

    data_path = os.path.join("data", params["data_file"])
    target_col = params["target"]
    features = params["features"]
    metric = params["metric"]
    k = params["k"]
    normalize_X = params["normalize_features"]
    normalize_y = params["normalize_target"]
    nulls_handling = params["nulls_handling"]
    rnn_seq_length = params["rnn_seq_length"]
    
    # Instancier un DataPrep
    dataprep = DataPrep(
        data_path, 
        features, 
        target_col, 
        nulls_handling, 
        normalize_X,
        normalize_y,
        rnn_seq_length
    )

    # Initier l'évaluation
    model_eval = ModelEval(
        dataprep, 
        model, 
        k,
        metric=metric,
        seed=seed
    )
    
    return model_eval
    

class DataPrep:
    """
    Préparateur de dataset
    """
    def __init__(
        self, 
        data_path, 
        cols_utiles,
        target_col,
        nulls_handling,
        normalize_X,
        normalize_y,
        rnn_seq_length
    ):
        # Initialiser les paramètres
        self.target_name = target_col
        self.cols_utiles = cols_utiles
        self.nulls_handling = nulls_handling
        self.normalize_X = normalize_X
        self.normalize_y = normalize_y
        self.rnn_seq_length = rnn_seq_length
        
        # Initialiser le dataset
        self.df = self.get_dataset(data_path)
        
    def get_dataset(self, data_path):
        """
        Dataset au format parquet ou csv
        """
        # Import des données
        if data_path.endswith(".parquet"):
            df_raw = pl.read_parquet(data_path)
        elif data_path.endswith(".csv"):
            df_raw = pl.read_csv(data_path)
        else:
            raise ValueError(f"Format du fichier {data_path} non supporté")
            
        # Garder uniquement les colonnes utiles, et les targets définis
        df_raw = df_raw.select(self.cols_utiles + [self.target_name])
        df_raw = df_raw.filter(pl.col(self.target_name).is_not_null())

        return df_raw
        
    def get_X_y(self, df):
        """
        Retourne features et target
        """
        X = df.select(
            [col for col in self.cols_utiles if col != self.target_name]
        )
        y = df.select(self.target_name)

        # Normaliser si nécessaire
        if self.normalize_X:
            X, _, _ = self.get_normalized_data(X)
        if self.normalize_y:
            y, self.y_mean, self.y_std = self.get_normalized_data(y)

        # Certaines librairies ne supportent pas les nulls
        if self.nulls_handling == "drop":
            X = X.drop_nulls()
        elif self.nulls_handling == "fill":
            X = X.fill_null(strategy="mean")
        
        # Chaque élément devient une liste de longueur rnn_seq_length
        if self.rnn_seq_length is not None:
            X = X.with_columns([
                pl.concat_list(
                    np.flip([pl.col(c).shift(i) for i in range(self.rnn_seq_length)])
                ).alias(c)
                for c in X.columns
            ])
            X = X.tail(-self.rnn_seq_length) # supprimer les 1ères lignes (nulls)
            y = y.tail(-self.rnn_seq_length)
        
        return X, y
    
    @staticmethod
    def get_normalized_data(df):
        """
        Mettre la moyenne du df target à 0, et sa variance à 1
        """
        df_std = df.with_columns(
            [((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
             .alias(col)
             for col in df.columns]
        )
        
        mean = df.mean()
        std = df.std()
        
        return df_std, mean, std
    
    
class ModelEval:
    def __init__(
        self, 
        dataprep,
        model,
        k,
        metric,
        seed
    ):
        """
        Préparation du dataset
        """
        # Initialisation des paramètres
        self.dataprep = dataprep
        self.model = model
        self.k = k
        self.metric = metric
        self.results = []
        
        data = self.dataprep.df
        
        # Extraire features et targets
        self.X, self.y = self.dataprep.get_X_y(data)
        
        # Mélanger les data points
        np.random.seed(seed)
        permutation = np.random.permutation(len(self.X))
        self.X, self.y = self.X[permutation], self.y[permutation]

        
    def get_scores(self):
        """
        Retourne la moyenne de k cross-validations du modèle pour la métrique
        choisie
        """
        scores = []
        slice_size = self.X.shape[0] // self.k
    
        # Itérer le calcul du score k fois
        for i in range(self.k):
            
            # Instancier une copie du modèle pour cette itération
            model = deepcopy(self.model)
            
            # Diviser le dataset en jeux de données d'entraînement/de test
            X_test = self.X.slice(i * slice_size, slice_size)
            y_test = self.y.slice(i * slice_size, slice_size)
            X_train = pl.concat(
                [self.X.slice(j * slice_size, slice_size) for j in range(self.k) 
                 if j != i]
            )
            y_train = pl.concat(
                [self.y.slice(j * slice_size, slice_size) for j in range(self.k) 
                 if j != i]
            )

            # Entraîner le modèle, effectuer une prédiction
            test_data = (X_test, y_test, self.target_name, self.metric)
            results = model.fit(X_train, y_train, test_data=test_data)
            self.results.append(results)
            y_pred = model.predict(X_test).ravel()
            
            # Dé-normaliser les données avant calcul du score si nécessaire
            if self.dataprep.normalize_y:
                y_pred = (y_pred * self.y_std + self.y_mean).ravel()
                y_test = y_test.with_columns(
                    (pl.col(self.target_name) * self.y_std + self.y_mean)
                    .alias(self.target_name)
                )
            
            # Calcul du score
            score = compute_score(y_pred, y_test, self.target_name, self.metric)
            scores.append(score)

        self.plot_results(self.results, self.metric)
            
        return np.array(scores)
    
    def report(self, scores):
        mean_score = round(scores.mean(), 4)
        logger.info(
            f"""{self.metric} moyen : {mean_score} \
            \nDétail des scores : {scores.tolist()}
            """
        )
        logger.shutdown()

    @staticmethod
    def plot_results(results, metric):
        data_records = []
        for run_idx, one_run in enumerate(results):
            for epoch_str, scores_dict in one_run.items():
                epoch_num = int(epoch_str.split()[1])
                train_score = scores_dict["train_score"]
                test_score = scores_dict["test_score"]
                data_records.append((run_idx, epoch_num, train_score, test_score))
        
        df = pl.DataFrame(
            data_records,
            schema=["run", "epoch", "train_score", "test_score"]
        )
        
        df_mean = (
            df.groupby("epoch", maintain_order=True)
            .agg([
                pl.col("train_score").mean().alias("train_score"),
                pl.col("test_score").mean().alias("test_score")
            ])
        )
        
        df_plot = df_mean.melt(
            id_vars="epoch",
            value_vars=["train_score", "test_score"],
            variable_name="type",
            value_name="score"
        )
        
        # Tracé seaborn
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_plot, x="epoch", y="score", hue="type", marker="o")
        plt.title("Score moyen par epoch")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(title="")
        plt.tight_layout()
        plt.show()
        
    @property
    def y_std(self):
        return self.dataprep.y_std
    
    @property
    def y_mean(self):
        return self.dataprep.y_mean
    
    @property
    def target_name(self):
        return self.dataprep.target_name
