import pandas as pd
import numpy as np
from prophet import Prophet

assign1_data_df = pd.read_csv('assignment_data_train.csv')
print(assign1_data_df.head())

assign1_data_df_keep = assign1_data_df[['Timestamp', 'trips']]
assign1_data_df_keep.Timestamp = pd.to_datetime(assign1_data_df_keep.Timestamp, infer_datetime_format=True)
assign1_data_df_keep = pd.DataFrame(assign1_data_df_keep.values, columns = ['ds','y'])
print(assign1_data_df_keep.head())

model = Prophet(changepoint_prior_scale=0.5)
model.fit(assign1_data_df_keep)

future = model.make_future_dataframe(periods=24*7, freq='h')
forecast = model.predict(future)

plot = model.plot(forecast)
components = model.plot_components(forecast)

modelFit = model