import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Iris Flower Prediction App')

st.write('This app predicts the *Iris flower* type!')

k_value = st.slider('Select the value of K for KNN', 1, 15)
st.write(f'You selected {k_value} as the value of K')

default_name = 'Iris'
flower_name = st.text_input('Enter the name of the flower', default_name)
st.write(f'The name of the flower is {flower_name}')

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

st.subheader('Iris Dataset')
st.dataframe(df)

X = df.drop('species', axis=1)
Y = df['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, Y_train)

st.subheader('Make A Prediction')
st.write('Enter the values below to make a prediction on the type of Iris flower!')

sepal_length = st.slider('Sepal Length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.slider('Sepal Width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.slider('Petal Length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.slider('Petal Width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

if st.button('Predict'):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    st.success(f'The Iris flower type is {prediction[0]}')

st.subheader('Model Accuracy')
Accuracy = model.score(X_test, Y_test)
st.write(f'The model accuracy is {Accuracy * 100:.2f}%')

st.subheader('Pairplot')

sns.pairplot(df, hue='species')
plt.show()

st.pyplot(plt)