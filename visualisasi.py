import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
def load_data(filename):
    """
    Memuat data dari file CSV
    :param filename: string, nama file CSV
    :return: tuple (features, labels)
    """
    try:
        data = pd.read_csv(filename, delimiter=';')
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' tidak ditemukan!")
    except pd.errors.ParserError:
        raise ValueError("Pastikan format delimiter sesuai!")

    # Pisahkan fitur dan label
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    """
    Menghitung jarak Euclidean antara dua titik
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Fungsi KNN tanpa Counter
def knn_classify(features, labels, test_point, k=5):
    """
    Klasifikasi KNN
    :param features: np.array, fitur data latih
    :param labels: np.array, label data latih
    :param test_point: np.array, titik yang diuji
    :param k: int, jumlah tetangga terdekat
    :return: label prediksi
    """
    distances = np.array([euclidean_distance(feature, test_point) for feature in features])
    nearest_indices = distances.argsort()[:k]
    nearest_labels = labels[nearest_indices]
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    most_common = unique_labels[np.argmax(counts)]
    return most_common

# Fungsi visualisasi Boxplot
def plot_boxplot(features, feature_names):
    """
    Membuat boxplot untuk fitur tertentu
    :param features: np.array, fitur data
    :param feature_names: list, nama-nama fitur
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(features, labels=feature_names, patch_artist=True)
    plt.title("Boxplot dari Fitur")
    plt.ylabel("Nilai Fitur")
    plt.grid()
    plt.show()

# Load data
filename = 'data iris.csv'
features, labels = load_data(filename)

# Visualisasi boxplot
feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
plot_boxplot(features, feature_names)
