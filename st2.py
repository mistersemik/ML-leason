import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

# Попытка импорта XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Попытка импорта TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def make_data(path='./dataset/cybersecurity_attacks.csv', Diagnostics=False, Statistics=False):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Attack Type Clean'] = df['Attack Type'].fillna('Normal').astype(str).str.strip()

    if Diagnostics:
        total_rows = len(df)
        attack_rows = (df['Attack Type Clean'].str.lower() != 'normal').sum()
        print(f"Всего строк: {total_rows}")
        print(f"Строк с атаками: {attack_rows} ({attack_rows / total_rows:.1%})")
        print("Уникальные значения 'Attack Type':")
        print(df['Attack Type Clean'].value_counts().head(10))

    df['is_attack'] = df['Attack Type Clean'].str.lower() != 'normal'
    daily = df.groupby(df['Timestamp'].dt.floor('D')).agg(y=('is_attack', 'sum')).reset_index()

    if Statistics:
        print(f"\nАгрегировано дней: {len(daily)}")
        print("Статистика по y (атак в день):")
        print(daily['y'].describe())
        print("Топ-5 дней по числу атак:")
        print(daily.nlargest(5, 'y'))

    daily.rename(columns={'Timestamp': 'ds'}, inplace=True)
    daily['unique_id'] = 'total'

    date_range = pd.date_range(start=daily['ds'].min(), end=daily['ds'].max(), freq='D')
    full_df = pd.DataFrame({'ds': date_range})
    full_df = pd.merge(full_df, daily[['ds', 'y']], on='ds', how='left')
    full_df['y'] = full_df['y'].fillna(0).astype(int)
    full_df['unique_id'] = 'total'
    return full_df


def detect_anomalies_with_if(df, contamination=0.05):
    from sklearn.ensemble import IsolationForest
    X = df[['y']].values
    clf = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_if'] = clf.fit_predict(X)
    return df


def train_autoencoder_anomaly_detector(df, encoding_dim=4):
    if not TF_AVAILABLE:
        print("TensorFlow не установлен. Пропуск автоэнкодера.")
        df['anomaly_ae'] = 0
        return df

    X = df[['y']].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    input_layer = Input(shape=(1,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(1, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    reconstruction_error = np.mean(np.abs(X_scaled - reconstructed), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    df['anomaly_ae'] = (reconstruction_error > threshold).astype(int)
    return df


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_lstm_model(train_df, test_df, lags=[1, 2, 3, 7, 14, 30]):
    if not TF_AVAILABLE:
        print("TensorFlow не установлен. Пропуск LSTM.")
        return np.full(len(test_df), np.nan)

    df = train_df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)

    feature_cols = [f'lag_{lag}' for lag in lags]
    X = df[feature_cols].values
    y = df['y'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    time_steps = 1
    X_train, y_train = create_dataset(X_scaled, y_scaled, time_steps)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(time_steps, len(feature_cols))),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Подготовка теста
    test_with_lags = test_df.copy()
    for lag in lags:
        test_with_lags[f'lag_{lag}'] = test_with_lags['y'].shift(lag)
    for lag in lags:
        if test_with_lags[f'lag_{lag}'].isna().any():
            last_val = train_df['y'].iloc[-lag] if lag <= len(train_df) else 0
            test_with_lags[f'lag_{lag}'].fillna(last_val, inplace=True)

    X_test = test_with_lags[feature_cols].values
    X_test_scaled = scaler_X.transform(X_test)
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], time_steps, len(feature_cols)))
    y_pred_scaled = model.predict(X_test_reshaped, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    return y_pred


def main(horizon=35, freq='D', lags=[1, 2, 3, 7, 14, 30], dayplot=60):
    df = make_data()
    print("Всего дней в данных:", len(df))

    # Обнаружение аномалий
    print("\n Обнаружение аномалий ")
    df = detect_anomalies_with_if(df)
    df = train_autoencoder_anomaly_detector(df)

    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()

    print(f"\nTrain: с {train['ds'].min()} по {train['ds'].max()} ({len(train)} дней)")
    print(f"Test:  с {test['ds'].min()} по {test['ds'].max()} ({len(test)} дней)")

    # MLForecast модели (прогноз)
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        LGBMRegressor(n_estimators=100, random_state=42),
        SVR(kernel='rbf', C=1.0, epsilon=0.1)
    ]
    if XGB_AVAILABLE:
        models.append(XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
        print("XGBoost добавлен.")
    else:
        print("XGBoost недоступен.")

    fcst = MLForecast(
        models=models,
        freq=freq,
        lags=lags,
        date_features=['dayofweek', 'dayofyear']
    )

    print("\nОбучение моделей...")
    fcst.fit(train, id_col='unique_id', time_col='ds', target_col='y')
    preds = fcst.predict(h=horizon)

    # LSTM
    print("\n Обучение LSTM")
    lstm_preds = train_lstm_model(train, test, lags)

    # Оценка
    mean_y = test['y'].mean()
    print(f"\nСреднее число атак в тесте: {mean_y:.2f}")
    print("\nРезультаты оценки моделей прогнозирования:")

    all_preds = {'LSTM': lstm_preds}
    for model in preds.columns[2:]:
        all_preds[model] = preds[model].values

    for name, pred in all_preds.items():
        if np.any(np.isnan(pred)):
            print(f"{name}: пропущено (не установлены зависимости)")
            continue
        mae = mean_absolute_error(test['y'], pred)
        rmse = np.sqrt(mean_squared_error(test['y'], pred))
        mae_pct = mae / mean_y * 100
        rmse_pct = rmse / mean_y * 100
        print(f"{name}: MAE = {mae:.2f} ({mae_pct:.1f}%), RMSE = {rmse:.2f} ({rmse_pct:.1f}%)")

    # Визуализация
    plt.figure(figsize=(14, 7))
    plot_hist = df.tail(dayplot)
    plt.plot(plot_hist['ds'], plot_hist['y'], label='История', color='black', marker='o', markersize=3)

    # Факт (тест)
    plt.plot(test['ds'], test['y'], label='Факт (тест)', color='red', linewidth=2, marker='o')

    # Прогнозы
    for model in preds.columns[2:]:
        plt.plot(preds['ds'], preds[model], label=f'Прогноз ({model})', linestyle='--', marker='x')
    if not np.any(np.isnan(lstm_preds)):
        plt.plot(test['ds'], lstm_preds, label='Прогноз (LSTM)', linestyle='--', color='green', marker='^')

    # Аномалии (опционально)
    anomalies_if = df[(df['anomaly_if'] == -1) & (df['ds'] >= plot_hist['ds'].min())]
    if not anomalies_if.empty:
        plt.scatter(anomalies_if['ds'], anomalies_if['y'], color='purple', label='Аномалии (Isolation Forest)', zorder=5)

    plt.title('Прогноз и аномалии: ежедневное число кибератак')
    plt.xlabel('Дата')
    plt.ylabel('Число атак')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
