from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import os

app = Flask(__name__)

# Print the current working directory and template folder path
print(f"Current working directory: {os.getcwd()}")
print(f"Template folder path: {os.path.join(os.getcwd(), 'templates')}")

# Generate toy time series data
def generate_data(n_periods=100):
    np.random.seed(42)
    t = np.arange(n_periods)
    trend = 0.1 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 1, n_periods)
    y = trend + seasonality + noise
    return pd.Series(y)

# Perform triple exponential smoothing
def triple_exponential_smoothing(data, alpha, beta, gamma):
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
    results = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
    return results.fittedvalues

# Perform ARIMA
def arima_model(data, p, d, q):
    model = ARIMA(data, order=(p, d, q))
    results = model.fit()
    return results.fittedvalues

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/arima')
def arima():
    return render_template('arima.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])
    gamma = float(request.form['gamma'])
    
    data = generate_data()
    fitted = triple_exponential_smoothing(data, alpha, beta, gamma)
    
    trace1 = go.Scatter(x=data.index.tolist(), y=data.values.tolist(), mode='lines', name='Original Data')
    trace2 = go.Scatter(x=fitted.index.tolist(), y=fitted.values.tolist(), mode='lines', name='Fitted Data')
    
    layout = go.Layout(title='Triple Exponential Smoothing',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title='Value'))
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    return jsonify(fig.to_dict())

@app.route('/update_arima', methods=['POST'])
def update_arima():
    p = int(request.form['p'])
    d = int(request.form['d'])
    q = int(request.form['q'])
    
    data = generate_data()
    fitted = arima_model(data, p, d, q)
    
    trace1 = go.Scatter(x=data.index.tolist(), y=data.values.tolist(), mode='lines', name='Original Data')
    trace2 = go.Scatter(x=fitted.index.tolist(), y=fitted.values.tolist(), mode='lines', name='Fitted Data')
    
    layout = go.Layout(title=f'ARIMA({p},{d},{q})',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title='Value'))
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    return jsonify(fig.to_dict())

if __name__ == '__main__':
    app.run(debug=True)