import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_data(path='./dataset/cybersecurity_attacks.csv', Diagnostics=False, Statistics=False):
    """
    Загружает и агрегирует данные кибератак по дням для последующего прогнозирования.

    Параметры:
    ----------
    path : str, по умолчанию './dataset/cybersecurity_attacks.csv'
        Путь к CSV-файлу с сырыми логами.
    Diagnostics : bool, по умолчанию False
        Если True — выводит информацию о распределении типов атак в исходных данных.
    Statistics : bool, по умолчанию False
        Если True — выводит описательную статистику по ежедневному числу атак (y).

    Возвращает:
    -----------
    pd.DataFrame
        Датафрейм с колонками: 'ds' (дата), 'y' (число атак), 'unique_id' (идентификатор ряда).
    """
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # ВСЕГДА создаём очищенную колонку — она нужна для логики
    df['Attack Type Clean'] = df['Attack Type'].fillna('Normal').astype(str).str.strip()

    # Диагностика — только по запросу
    if Diagnostics:
        total_rows = len(df)
        attack_rows = (df['Attack Type Clean'].str.lower() != 'normal').sum()
        print(f"Всего строк: {total_rows}")
        print(f"Строк с атаками: {attack_rows} ({attack_rows / total_rows:.1%})")
        print("Уникальные значения 'Attack Type':")
        print(df['Attack Type Clean'].value_counts().head(10))

    # Теперь безопасно использовать 'Attack Type Clean'
    df['is_attack'] = df['Attack Type Clean'].str.lower() != 'normal'

    # Агрегация по дням
    daily = df.groupby(df['Timestamp'].dt.floor('D')).agg(
        y=('is_attack', 'sum')
    ).reset_index()

    if Statistics:
        print(f"\nАгрегировано дней: {len(daily)}")
        print("Статистика по y (атак в день):")
        print(daily['y'].describe())
        print("Топ-5 дней по числу атак:")
        print(daily.nlargest(5, 'y'))

    daily.rename(columns={'Timestamp': 'ds'}, inplace=True)
    daily['unique_id'] = 'total'

    # Полная временная сетка (без пропусков)
    date_range = pd.date_range(start=daily['ds'].min(), end=daily['ds'].max(), freq='D')
    full_df = pd.DataFrame({'ds': date_range})
    full_df = pd.merge(full_df, daily[['ds', 'y']], on='ds', how='left')
    full_df['y'] = full_df['y'].fillna(0).astype(int)
    full_df['unique_id'] = 'total'

    return full_df


def main(horizon=35, freq='D', lags=[1, 2, 3, 7, 14, 30], dayplot=60):
    """
    Обучает и оценивает модели машинного обучения для прогнозирования количества кибератак во времени.

    Параметры:
    ----------
    horizon : int, по умолчанию 35
        Горизонт прогнозирования в единицах времени (например, дней).
        Последние `horizon` наблюдений выделяются в тестовую выборку.

    freq : str, по умолчанию 'D'
        Частота временного ряда (например, 'D' — ежедневно, 'H' — почасово).
        Должна соответствовать частоте агрегации в данных.

    lags : list of int, по умолчанию [1, 2, 3, 7, 14, 30]
        Список лагов (сдвигов во времени), используемых как признаки.
        Например, лаг 7 означает использование значения ряда 7 дней назад.

    dayplot : int, по умолчанию 60
        Количество последних дней, отображаемых на графике для наглядности.

    Модели:
    -------
    - LightGBM (LGBMRegressor)
    - Random Forest (RandomForestRegressor)
    - XGBoost (XGBRegressor, если установлен)

    Возвращает:
    -----------
    None. Выводит метрики качества и график прогноза vs факта.
    """
    df = make_data()
    print("Всего дней в данных:", len(df))

    # === Разделение на train и test ===
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()

    print(f"\nTrain: с {train['ds'].min()} по {train['ds'].max()} ({len(train)} дней)")
    print(f"Test:  с {test['ds'].min()} по {test['ds'].max()} ({len(test)} дней)")

    # === Подготовка моделей ===
    models = [
        LGBMRegressor(n_estimators=100, random_state=42),
        RandomForestRegressor(n_estimators=100, random_state=42)
    ]

    # Попытка добавить XGBoost, если установлен
    try:
        from xgboost import XGBRegressor
        models.append(XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
        print("XGBoost доступен и добавлен в сравнение.")
    except ImportError:
        print("XGBoost не установлен. Продолжаем без него.")

    # === Обучение ===
    fcst = MLForecast(
        models=models,
        freq=freq,
        lags=lags,
        date_features=['dayofweek', 'dayofyear']
    )

    print("\nОбучение на train...")
    fcst.fit(train, id_col='unique_id', time_col='ds', target_col='y')

    # === Прогноз ===
    preds = fcst.predict(h=horizon)

    # === Оценка качества ===
    print("\nОценка моделей на исторических данных (test):")
    for model in preds.columns[2:]:
        mae = mean_absolute_error(test['y'], preds[model])
        rmse = np.sqrt(mean_squared_error(test['y'], preds[model]))
        mean_y = test['y'].mean()
        mae_pct = mae / mean_y * 100
        rmse_pct = rmse / mean_y * 100
        print(f"{model}: MAE = {mae:.2f} ({mae_pct:.1f}%), RMSE = {rmse:.2f} ({rmse_pct:.1f}%)")

    # === Визуализация ===
    plt.figure(figsize=(12, 6))
    plot_hist = df.tail(dayplot)
    plt.plot(plot_hist['ds'], plot_hist['y'], label='История', color='black', marker='o', markersize=3)
    plt.plot(test['ds'], test['y'], label='Факт (тест)', color='red', linewidth=2, marker='o')

    for model in preds.columns[2:]:
        plt.plot(preds['ds'], preds[model], label=f'Прогноз ({model})', linestyle='--', marker='x')

    plt.title('Прогноз vs Факт: количество кибератак')
    plt.xlabel('Дата')
    plt.ylabel('Количество атак')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
