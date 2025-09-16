import pandas as pd
import numpy as np
import prophet

assign1_data_df = pd.read_csv('assignment_data_train.csv')

assign1_data_df_keep = assign1_data_df[['Timestamp', 'trips']]
assign1_data_df_keep.Timestamp = pd.to_datetime(assign1_data_df_keep.Timestamp, infer_datetime_format=True)
assign1_data_df_keep = pd.DataFrame(assign1_data_df_keep.values, columns = ['ds','y'])

model = prophet.Prophet(changepoint_prior_scale=0.5)
modelFit = model.fit(assign1_data_df_keep)

future = modelFit.make_future_dataframe(periods=744, freq='h', include_history=False)
forecast_test = modelFit.predict(future)

# change pred to np array
pred = np.array(forecast_test['yhat'])
# round pred to nearest integer
pred = np.round(pred).astype(float)