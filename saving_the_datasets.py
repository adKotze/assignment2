# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd

# %% [markdown]
# # 1. Function approximation 
# $f(x, y) = x^2 + y$

# %%
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x, y)
zz = xx**2 + yy

fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis')])

# Update layout for better visualization
fig.update_layout(
    title='3D Surface plot of z = x^2 + y',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# Show the plot
fig.show()

# %%
x = np.random.permutation(np.linspace(-10, 10, 50000))
y = np.random.permutation(np.linspace(-10, 10, 50000))
z = x**2 + y
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8))])

# Update layout for better visualization
fig.update_layout(
    title='3D Scatter Plot of z = x^2 + y',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Show the plot
fig.show()

# %%
# saving the data
df1 = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z
})
df1.to_csv('Function1.csv')

# %%
df1.shape

# %%
df1.head()

# %% [markdown]
# # Function 2
# 
# $f(x, y) = e^{-\frac{1}{100}(x+y-xy)^2}$

# %%
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x, y)
zz = np.exp(-(xx + yy - xx*yy)**2/100)

fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis')])

# Update layout for better visualization
fig.update_layout(
    title='3D Surface plot of z = sin(x) + sin(y) + x*y',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# Show the plot
fig.show()

# %%
x = np.random.permutation(np.linspace(-10, 10, 50000))
y = np.random.permutation(np.linspace(-10, 10, 50000))
z = np.exp(-(x + y - x*y)**2/100)
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8))])

# Update layout for better visualization
fig.update_layout(
    title='3D Scatter Plot of z = x^2 + y',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Show the plot
fig.show()

# %%
# saving the data
df2 = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z
})
df2.to_csv('Function2.csv')

# %%
df2.head()

# %%
df2.shape

# %% [markdown]
# # Function 3 
# 
# $f(x, y) = 10sin(x)cos(y) + \frac{1}{10}x^2 + \frac{1}{10}y^2$

# %%
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x, y)
zz = 10*np.sin(xx)*np.cos(yy) + xx**2/10 + yy**2/10

fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis')])

# Update layout for better visualization
fig.update_layout(
    title='3D Surface plot of z = sin(x^2 + y^2)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# Show the plot
fig.show()

# %%
x = np.random.permutation(np.linspace(-10, 10, 50000))
y = np.random.permutation(np.linspace(-10, 10, 50000))
z = 10*np.sin(x)*np.cos(y) + x**2/10 + y**2/10 #+ np.random.normal(0, 0.01, x.shape) # 
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8))])

# Update layout for better visualization
fig.update_layout(
    title='3D Scatter Plot Example',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Show the plot
fig.show()

# %%
# saving the data
df3 = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z
})
df3.to_csv('Function3.csv')

# %%
df3.head()

# %%
df3.shape

# %% [markdown]
# # The classification dataset

# %% [markdown]
# ## 1. Iris Dataset

# %%
from sklearn.datasets import load_iris

# %%
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# %%
df_iris.head()

# %%
df_iris.shape

# %%
df_iris.describe().T

# %%
df_iris.to_csv('Iris.csv')

# %%
df_iris['target'].value_counts()

# %% [markdown]
# ## 2. MNIST Dataset

# %%
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %% [markdown]
# # 3. CIFAR Dataset

# %%
# from tensorflow.keras.datasets import cifar10
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# %%



