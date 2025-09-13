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

pred = forecast_test['yhat'].tolist()
pred = [round(num) for num in pred]

pred = [int(num) for num in pred]

'''def checkNumbers(pred):
    for i in pred:
        if not isinstance(i, (float, int)):
            print(f'False at {i} which is a {type(i)}')
    print(f'True for all {len(pred)} values')

def test_valid_pred(pred):
  assert (len(np.squeeze(pred))==744 and checkNumbers(np.squeeze(pred))), \
    "Make sure your prediction consists of integers\nor floating point numbers, and is a list or array of 744\nfuture predictions!"


print('Length of pred:', len(pred))
checkNumbers(pred)

test_valid_pred(pred)'''