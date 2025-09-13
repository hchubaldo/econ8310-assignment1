import pandas as pd
import numpy as np
import prophet

assign1_data_df = pd.read_csv('assignment_data_train.csv')

assign1_data_df_keep = assign1_data_df[['Timestamp', 'trips']]
assign1_data_df_keep.Timestamp = pd.to_datetime(assign1_data_df_keep.Timestamp, infer_datetime_format=True)
assign1_data_df_keep = pd.DataFrame(assign1_data_df_keep.values, columns = ['ds','y'])

model = prophet.Prophet(changepoint_prior_scale=0.5)
modelFit = model.fit(assign1_data_df_keep)

assign1_test_data_df = pd.read_csv('assignment_data_test.csv')
assign1_test_data_df_keep = assign1_test_data_df[['Timestamp']]
assign1_test_data_df_keep.Timestamp = pd.to_datetime(assign1_test_data_df_keep.Timestamp, infer_datetime_format=True)
assign1_test_data_df_keep = pd.DataFrame(assign1_test_data_df_keep.values, columns = ['ds'])
forecast_test = modelFit.predict(assign1_test_data_df_keep)

pred = forecast_test['yhat'].tolist()

pred = [round(num) for num in pred]
print(pred)