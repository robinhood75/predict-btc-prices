import numpy as np
from src.utils import compute_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BaseRNN(nn.Module):
    """
    Classe de base pour l'implémentation de variantes de LSTM
    """
    def __init__(self, classif=False):
        """
        @param dim_fc : si None, il n'y a pas de FC au milieu du réseau
        @param classif : bool tâche de classification
        """
        super().__init__()
        self.classif = classif

    def forward(self, input):
        raise NotImplementedError
    
    def fit(self, X_train, y_train, epochs=40, batch_size=32, test_data=None):    
        """
        @param test_data: si pas None, tester le modèle pendant l'entraînement et sauvegarder
        les résultats dans self.results. Format suivant : (X_test, y_test, target_name, metric)
        ()
        """   
        if self.classif:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        X_train_torch = self.get_torch_from_polars(X_train)
        y_train_torch = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results = {}

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                optimizer.step()

            # Monitorer les progrès du modèle
            if test_data is not None and epoch % 3== 0:
                results[f"epoch {epoch}"] = {}
                res = results[f"epoch {epoch}"]
                X_test, y_test, target_name, metric = test_data

                train_pred = self.predict(X_train).ravel()
                score = compute_score(train_pred, y_train, target_name, metric)
                res["train_score"] = score

                test_pred = self.predict(X_test).ravel()
                score = compute_score(test_pred, y_test, target_name, metric)
                res["test_score"] = score

                self.train()

        return results

    def predict(self, X_test):
        X_test = self.get_torch_from_polars(X_test)

        self.eval()
        with torch.no_grad():
            ret_torch = self.forward(X_test)
            return ret_torch.numpy()
        
    @staticmethod
    def get_torch_from_polars(X):
        """
        X est un pl.Dataframe de dimensions (N, D) avec chaque élément de longueur T
        A transformer en np.array de dimensions (N, T, D) ("batch first")
        """
        columns = [X[col].to_list() for col in X.columns]
        X = np.stack(columns, axis=-1)
        X = torch.tensor(X, dtype=torch.float32)
        
        return X
    

class LSTMSimpleNet(BaseRNN):
    def __init__(self, dim_input, dim_recurrent=32, num_layers=1, classif=False):
        """
        Architecture :
        - LSTM : dim_input -> dim_recurrent
        - ReLU
        - Dropout
        - FC : dim_recurrent -> 1
        """
        super().__init__(classif)
        self.lstm = nn.LSTM(
            input_size = dim_input, 
            hidden_size = dim_recurrent, 
            num_layers = num_layers,
            batch_first=False
        )
        self.fc_out = nn.Linear(dim_recurrent, 1)

    def forward(self, input):
        output, _ = self.lstm(input.permute(1, 0, 2))
        output = F.dropout1d(output[-1], p=0.)
        return self.fc_out(F.relu(output))        


class CNNRNNTorch(BaseRNN):
    def __init__(
            self, 
            dim_input, 
            n_channels=16, 
            kernel_size=5, 
            dim_recurrent=64, 
            num_layers=1, 
            dim_fc=None, 
            classif=False
        ):
        """
        Architecture :
        - Conv1d : (dim_input, L) -> (n_channels, L - kernel_size + 1)
        - ReLU
        - LSTM : n_channels -> dim_recurrent
        - ReLU
        - FC : dim_recurrent -> dim_fc
        - BatchNorm
        - Dropout
        - ReLU
        - FC : dim_fc -> 1

        @param dim_fc: si None, seul 1 FC layer est présent en fin de chaîne et non 2
        """
        super().__init__(classif=classif)
        self.conv = nn.Conv1d(dim_input, n_channels, kernel_size=kernel_size)
        self.lstm = nn.LSTM(
            input_size = n_channels, 
            hidden_size = dim_recurrent, 
            num_layers = num_layers,
            batch_first=False
        )
        if dim_fc is not None:
            self.fc = nn.Linear(dim_recurrent, dim_fc)
            self.batch_norm = nn.BatchNorm1d(dim_fc)
            self.fc_out = nn.Linear(dim_fc, 1)
        else:
            self.fc_out = nn.Linear(dim_recurrent, 1)

    def forward(self, input):
        output = input.permute(0, 2, 1)
        output = F.relu(self.conv(output))
        output, _ = self.lstm(output.permute(2, 0, 1))
        output = output[-1]
        if hasattr(self, "fc"):
            output = self.fc(F.relu(output))
            output = self.batch_norm(output)
            output = F.dropout(output, p=0.1)
        output = self.fc_out(F.relu(output))
        return output


class EnsembleRNN(BaseRNN):
    """
    Ce modèle aggrège les outputs de modèles spécifiques à certaines tâches
    """
    def __init__(self, models, features_decoupage, classif=False, bagging_method="averaging"):
        """
        @param features_decoupage : liste de listes de features, dont chacune 
        correspond aux features pris en entrée par le modèle correspondant
        """
        super().__init__(classif)
        self.models = models
        self.features_decoupage = features_decoupage
        self.bagging_method = bagging_method
        self.fc_out = nn.Linear(len(models), 1)

    def forward(self, input):
        inputs = [input[:, :, indices] for indices in self.features_decoupage]
        representations = []
        for model, input in zip(self.models, inputs):
            ret = model.forward(input)
            representations.append(ret)
        output = torch.cat(representations, dim=-1)

        if self.bagging_method == "averaging":
            output = output.mean(dim=-1, keepdim=True)
        elif self.bagging_method == "fc":
            output = self.fc_out(F.relu(output))
        else:
            raise ValueError(f"Unknown bagging method {self.bagging_method}")
        
        return output
