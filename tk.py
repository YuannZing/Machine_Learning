import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np

# Fungsi untuk memuat data
def load_data(filename):
    data = pd.read_csv(filename, delimiter=';')
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Fungsi KNN tanpa Counter
def knn_classify(features, labels, test_point, k=5):
    distances = np.array([euclidean_distance(feature, test_point) for feature in features])
    nearest_indices = distances.argsort()[:k]
    nearest_labels = labels[nearest_indices]
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    most_common = unique_labels[np.argmax(counts)]
    return most_common

# Fungsi untuk tombol prediksi
def on_predict_click():
    try:
        # Mengambil nilai dari input pengguna
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        test_point = np.array([sepal_length, sepal_width, petal_length, petal_width])
        predicted_class = knn_classify(features, labels, test_point, k=5)
        
        # Menampilkan hasil prediksi
        result_label.config(text=f"Hasil: {predicted_class}")
    except ValueError:
        messagebox.showerror("Kesalahan Input", "Harap masukkan angka yang valid untuk semua input!")

# Load data
filename = 'data iris.csv' 
features, labels = load_data(filename)

# Membuat GUI dengan Tkinter
root = tk.Tk()
root.title("Prediksi Kelas Iris")
root.geometry("250x300")

# Label Judul
title_label = tk.Label(root, text="Input Data", font=("Arial", 14, "bold"))
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Input Sepal Length
label_sepal_length = tk.Label(root, text="Sepal Length :")
label_sepal_length.grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_sepal_length = tk.Entry(root)
entry_sepal_length.grid(row=1, column=1, padx=5, pady=5)

# Input Sepal Width
label_sepal_width = tk.Label(root, text="Sepal Width :")
label_sepal_width.grid(row=2, column=0, sticky="e", padx=5, pady=5)
entry_sepal_width = tk.Entry(root)
entry_sepal_width.grid(row=2, column=1, padx=5, pady=5)

# Input Petal Length
label_petal_length = tk.Label(root, text="Petal Length :")
label_petal_length.grid(row=3, column=0, sticky="e", padx=5, pady=5)
entry_petal_length = tk.Entry(root)
entry_petal_length.grid(row=3, column=1, padx=5, pady=5)

# Input Petal Width
label_petal_width = tk.Label(root, text="Petal Width :")
label_petal_width.grid(row=4, column=0, sticky="e", padx=5, pady=5)
entry_petal_width = tk.Entry(root)
entry_petal_width.grid(row=4, column=1, padx=5, pady=5)

# Tombol Prediksi
predict_button = tk.Button(root, text="Prediksi", command=on_predict_click)
predict_button.grid(row=5, column=0, columnspan=2, pady=10)

# Label Hasil
result_label = tk.Label(root, text="Hasil: ", font=("Arial", 12))
result_label.grid(row=6, column=0, columnspan=2, pady=10)

# Menjalankan aplikasi
root.mainloop()
