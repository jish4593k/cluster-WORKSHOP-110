import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim


def load_data():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select CSV File")
    return pd.read_csv(file_path, names=['x1', 'x2', 'x3', 'x4', 'class_name'])


def perform_kmeans(df):
    X = df.iloc[:, [0, 1, 2, 3]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['predicted_class'] = kmeans.fit_predict(X)
    return df, kmeans.cluster_centers_


def linear_regression(df):
    X = torch.tensor(df['x1'].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df['x2'].values, dtype=torch.float32).view(-1, 1)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    df['predicted_x2'] = model(X).detach().numpy()
    return df


df = load_data()

df_kmeans, centroids = perform_kmeans(df)

df_regression = linear_regression(df)


fig = px.scatter(df_kmeans, x='x1', y='x2', color='predicted_class', title='KMeans Clustering')
fig.add_trace(px.scatter(x=centroids[:, 0], y=centroids[:, 1], color=[0, 1, 2], labels={'0': 'Centroids'}).data[0])
fig.show()


fig = px.scatter(df_regression, x='x1', y='x2', title='Linear Regression')
fig.add_trace(px.line(df_regression, x='x1', y='predicted_x2', labels={'predicted_x2': 'Regression Line'}).data[0])
fig.show()
