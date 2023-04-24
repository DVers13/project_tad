import pickle

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import streamlit as st
import tensorflow as tf

def main():
    model = tf.keras.models.load_model('model_dumps/neural')
    test_data = load_test_data("data/X.csv")
    y_data = load_test_data("data/y.csv")
    
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""О наборе данных
Контекст
Набор данных собран с различных веб-ресурсов с целью изучения рынка подержанных автомобилей и попытки построить модель, которая эффективно прогнозирует цену автомобиля на основе его параметров (как числовых, так и категориальных).

Содержание
Данные были собраны в Беларуси (Западная Европа) 2 декабря 2019 года, поэтому набор данных довольно свежий и актуальный.""")

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:

* transmission – Type of the transmission,
* color – Body color,
* odometer_value – Odometer state in kilometers,
* year_produced – The year the car has been produced.,
* engine_fuel – Fuel type of the engine,
* engine_has_gas – Is the car equipped with propane tank and tubing?,
* engine_type – Engine type,
* engine_capacity – The capacity of the engine in liters, numerical column,
* body_type – Type of the body (hatchback, sedan, etc.,
* has_warranty – Does the car have warranty?,
* state – New/owned/emergency. Emergency means the car has been damaged, sometimes severely,
* drivetrain - Front/rear/all drivetrain, categorical column.
* is_exchangeable - If is_exchangeable is True the owner of the car is ready to exchange this car to other cars with little or no additional payment.
* location_region - Categorical column, location_region is a region in Belarus where the car is listed for sale.,
* number_of_photos - Number of photos the car has. numerical
* up_counter - Number of times the car has been upped, numerical.
* feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9 - Is the option like alloy wheels, conditioner, etc. is present in the car.
* duration_listed - Number of days the car is listed in the catalog.

К категориальным признакам относятся:
*    "color",
*    "engine_fuel",
*    "engine_type",
*    "body_type",
*    "state",
*    "drivetrain",

К порядковым признакам относятся:
*   transmission
*   odometer_value
*   year_produced
*   engine_capacity
*   location_region
*   number_of_photos
*   up_counter
*   duration_listed

К бинарным признакам относятся:
*   "engine_has_gas",
*   "has_warranty",
*   "is_exchangeable",
*   "feature_0",
*   "feature_1",
*   "feature_2",
*   "feature_3",
*   "feature_4",
*   "feature_5",
*   "feature_6",
*   "feature_7",
*   "feature_8",
*   "feature_9"
""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["Метрики", "Первые 5 предсказанных значений"]
        )

        if request == "Метрики":
            st.header("R^2")
            rt = r2_score(model.predict(test_data), y_data)
            st.write(f"{rt}")
            st.header("MAE")
            mae = mean_absolute_error(model.predict(test_data), y_data)
            st.write(f"{mae}")
        elif request == "Первые 5 предсказанных значений":
            st.header("Первые 5 предсказанных значений")
            first_5_test = test_data.iloc[:5, :]
            first_5_pred = model.predict(first_5_test)
            for item in first_5_pred:
                item = float(item)
                st.write(round(item))


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


if __name__ == '__main__':
    main()
