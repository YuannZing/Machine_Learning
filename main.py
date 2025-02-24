import pandas as pd
import numpy as np


# Membaca data dari file CSV menggunakan pandas
def load_data(filename):
    data = pd.read_csv(filename, delimiter=';')
    features = data.iloc[:, :-1].values  # Semua kolom kecuali yang terakhir sebagai fitur
    labels = data.iloc[:, -1].values     # Kolom terakhir sebagai label
    return features, labels


# Fungsi untuk menghitung jarak Euclidean antara dua titik
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Fungsi KNN tanpa Counter
def knn_classify(features, labels, test_point, k=3):
    # Menghitung jarak setiap titik di data latih ke titik uji
    distances = np.array([euclidean_distance(feature, test_point) for feature in features])
    # Mengurutkan berdasarkan jarak terdekat dan memilih K terdekat
    nearest_indices = distances.argsort()[:k]
    nearest_labels = labels[nearest_indices]
    
    # Menemukan kelas yang paling sering muncul di antara tetangga terdekat
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    most_common = unique_labels[np.argmax(counts)]
    return most_common


# Contoh penggunaan
filename = 'data iris 1.csv'  # Ganti dengan nama file CSV Anda
features, labels = load_data(filename)


# Tentukan titik uji
test_point = np.array([6, 3, 4.8, 1.8])  # Contoh titik uji


# Prediksi kelas dengan K=3
predicted_class = knn_classify(features, labels, test_point, k=5)
print(f"Kelas yang diprediksi untuk titik uji {test_point} adalah: {predicted_class}")


