import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import skops.io as sio
import os
import matplotlib.pyplot as plt

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Usuwanie kolumn, które nie będą używane
    df = df.drop(['appid', 'english', 'steamspy_tags'], axis=1)

    # Usuwanie rekordów z zerowymi wartościami w ważnych kolumnach
    df = df[(df['average_playtime'] != 0) & (df['median_playtime'] != 0) & (df['price'] != 0)]

    # Obcięcie cen do 10%-90% percentyla
    df = df[(df['price'] >= df['price'].quantile(0.1)) & (df['price'] <= df['price'].quantile(0.9))]

    # Mapowanie kolumny 'owners'
    mapping = {
        '0-20000': 1,
        '20000-50000': 2,
        '50000-100000': 3,
        '100000-200000': 4,
        '200000-500000': 5,
        '500000-1000000': 6,
        '1000000-2000000': 7,
        '2000000-5000000': 8,
        '5000000-10000000': 9
    }
    df['owners'] = df['owners'].map(mapping)

    # Dodanie cech boolowskich
    df['is_multiplatform'] = df['platforms'].str.contains(';')
    df['is_Single-Player'] = df['categories'].str.contains('Single-player')
    df['is_Action'] = df['genres'].str.contains('Action')

    # Konwersja bool na int
    bool_cols = ['is_multiplatform', 'is_Single-Player', 'is_Action']
    df[bool_cols] = df[bool_cols].astype(int)

    # Kolumny cech
    feature_cols = ['positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime',
                    'owners', 'is_multiplatform', 'is_Single-Player', 'is_Action']

    # Usunięcie braków danych
    df = df.dropna(subset=feature_cols + ['price'])

    X = df[feature_cols]
    y = df['price']

    return X, y

def create_pipeline():
    numeric_features = ['positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'owners']
    categorical_features = ['is_multiplatform', 'is_Single-Player', 'is_Action']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return pipeline

def main():
    X, y = load_and_preprocess_data("Data/steam.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"Model score (R^2) na danych testowych: {score:.3f}")

    # Predykcja
    y_pred = pipeline.predict(X_test)

    # Metryki regresji
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    # Zapis metryk do pliku
    with open("Results/metrics.txt", "w") as f:
        f.write(f"R2 Score: {round(score, 3)}\n")
        f.write(f"MAE: {round(mae, 2)}\n")
        f.write(f"RMSE: {round(rmse, 2)}\n")

    # Wykres rzeczywista vs przewidywana cena
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Rzeczywista cena")
    plt.ylabel("Przewidywana cena")
    plt.title("Rzeczywista vs Przewidywana cena gry")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Results/model_results.png", dpi=120)

    # Zapis modelu
    os.makedirs("Model", exist_ok=True)
    sio.dump(pipeline, "Model/steam_pipeline.skops")
    print("Model saved to Model/steam_pipeline.skops")

if __name__ == "__main__":
    main()