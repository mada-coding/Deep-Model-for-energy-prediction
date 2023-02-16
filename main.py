import pandas as pd
import plotly.express as px

path_csv = F"/content/drive/My Drive/grafic.csv"
data = pd.read_csv(path_csv)

fig = px.line(data, x = 'Data', y = 'Consum', title='Consumul in timp')
fig.show()

fig1 = px.line(data, x = 'Data', y = 'Productie', title='Productia in timp')
fig1.show()

fig2 = px.line(data, x = 'Data', y = 'Carbune', title='Productia de carbune in timp')
fig2.show()

fig3 = px.line(data, x = 'Data', y = 'Hidrocarburi', title='Productia de Hidrocarburi in timp')
fig3.show()

fig4 = px.line(data, x = 'Data', y = 'Ape', title='Productia de Ape in timp')
fig4.show()

fig5 = px.line(data, x = 'Data', y = 'Nuclear', title='Productia de Nucleara in timp')
fig5.show()

fig6 = px.line(data, x = 'Data', y = 'Eolian', title='Productia de Eolian in timp')
fig6.show()

fig7 = px.line(data, x = 'Data', y = 'Foto', title='Productia de Foto in timp')
fig7.show()

fig8 = px.line(data, x = 'Data', y = 'Biomasa', title='Productia de Biomasa in timp')
fig8.show()


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from datetime import datetime, timedelta
import pandas as pd 

import yaml
import os

with open(f'{os.getcwd()}/conf.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

path_csv = F"/content/drive/My Drive/grafic.csv"
d = pd.read_csv(path_csv)
d['Data'] = [datetime.strptime(x, '%d-%m-%Y %H:%M:%S') for x in d['Data']]

d = d.groupby('Data', as_index=False)['Biomasa'].mean()
d.sort_values('Data', inplace=True)

deep_learner = DeepModelTS(
    data=d, 
    Y_var='Biomasa',
    lag=conf.get('lag'),
    LSTM_layer_depth=conf.get('LSTM_layer_depth'),
    epochs=conf.get('epochs'),
    train_test_split=conf.get('train_test_split')
)

model = deep_learner.LSTModel()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
yhat = deep_learner.predict()

if len(yhat) > 0:

    fc = d.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    plt.figure(figsize=(12, 8))
    for dtype in ['Biomasa', 'forecast']:
        plt.plot(
            'Data',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()   
    

deep_learner = DeepModelTS(
    data=d, 
    Y_var='Biomasa',
    lag=24,
    LSTM_layer_depth=64,
    epochs=20,
    train_test_split=0 
)

deep_learner.LSTModel()

n_ahead = 720
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

fc = d.tail(400).copy() 
fc['type'] = 'original'

last_date = max(fc['Data'])
hat_frame = pd.DataFrame({
    'Data': [last_date + timedelta(hours=x + 1) for x in range(n_ahead)], 
    'Biomasa': yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

plt.figure(figsize=(12, 8))
for col_type in ['original', 'forecast']:
    plt.plot(
        'Data', 
        'Biomasa', 
        data=fc[fc['type']==col_type],
        label=col_type
        )

plt.legend()
plt.grid()
plt.show()    
