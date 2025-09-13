import pandas as pd
import numpy as np
import prophet

assign1_data_df = pd.read_csv('assignment_data_train.csv')
print(assign1_data_df.head())

assign1_data_df_keep = assign1_data_df[['Timestamp', 'trips']]
assign1_data_df_keep.Timestamp = pd.to_datetime(assign1_data_df_keep.Timestamp, infer_datetime_format=True)
assign1_data_df_keep = pd.DataFrame(assign1_data_df_keep.values, columns = ['ds','y'])
print(assign1_data_df_keep.head())

model = prophet.Prophet(changepoint_prior_scale=0.5)
modelFit = model.fit(assign1_data_df_keep)

future = modelFit.make_future_dataframe(periods=24*7, freq='h')
forecast = modelFit.predict(future)

modelFit.plot(forecast)
modelFit.plot_components(forecast)

assign1_test_data_df = pd.read_csv('assignment_data_test.csv')
assign1_test_data_df_keep = assign1_test_data_df[['Timestamp']]
assign1_test_data_df_keep.Timestamp = pd.to_datetime(assign1_test_data_df_keep.Timestamp, infer_datetime_format=True)
assign1_test_data_df_keep = pd.DataFrame(assign1_test_data_df_keep.values, columns = ['ds'])
print(assign1_test_data_df_keep.head())
forecast_test = modelFit.predict(assign1_test_data_df_keep)
print(forecast_test[['ds', 'yhat']].head())

pred = forecast_test['yhat'].tolist()

pred = [round(num) for num in pred]
print(type(pred[0]))