import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

# Fungsi untuk memuat data
def load_data(filename):
    data = pd.read_csv(filename, delimiter=';')
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels

# Fungsi untuk membagi data secara random
def split_data(features, labels, test_size=0.2):
    data = list(zip(features, labels))
    random.shuffle(data)
    features, labels = zip(*data)
    
    split_index = int(len(features) * (1 - test_size))
    train_features = np.array(features[:split_index])
    train_labels = np.array(labels[:split_index])
    test_features = np.array(features[split_index:])
    test_labels = np.array(labels[split_index:])
    
    return train_features, train_labels, test_features, test_labels

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Fungsi KNN
def knn_classify(features, labels, test_point, k=5):
    distances = np.array([euclidean_distance(feature, test_point) for feature in features])
    nearest_indices = distances.argsort()[:k]
    nearest_labels = labels[nearest_indices]
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    most_common = unique_labels[np.argmax(counts)]
    return most_common

# Fungsi untuk menghitung akurasi
def calculate_accuracy(features, labels, test_features, test_labels, k):
    correct_predictions = 0
    for i in range(len(test_features)):
        prediction = knn_classify(features, labels, test_features[i], k)
        if prediction == test_labels[i]:
            correct_predictions += 1
    return correct_predictions / len(test_labels)

# Fungsi untuk mencari nilai K terbaik dengan k-fold cross-validation
def find_best_k_kfold(features, labels, max_k=20, n_splits=5):
    k_values = range(1, max_k + 1)
    accuracies = []

    for k in k_values:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        for train_index, test_index in kf.split(features):
            train_features, test_features = features[train_index], features[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            accuracy = calculate_accuracy(train_features, train_labels, test_features, test_labels, k)
            fold_accuracies.append(accuracy)
        average_accuracy = np.mean(fold_accuracies)
        accuracies.append(average_accuracy)

    best_k = k_values[np.argmax(accuracies)]
    best_accuracy = max(accuracies)

    # Visualisasi grafik akurasi
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Akurasi K-Fold Cross-Validation terhadap Nilai K', fontsize=14)
    plt.xlabel('Nilai K', fontsize=12)
    plt.ylabel('Akurasi Rata-rata', fontsize=12)
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()

    return best_k, best_accuracy

# Fungsi untuk membuat dan menampilkan confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Fungsi untuk tombol prediksi
def on_predict_click():
    try:
        # Mengambil nilai dari input pengguna
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        test_point = np.array([sepal_length, sepal_width, petal_length, petal_width])
        predicted_class = knn_classify(train_features, train_labels, test_point, k=best_k)
        
        # Menampilkan hasil prediksi
        result_label.config(text=f"Hasil: {predicted_class}")
    except ValueError:
        messagebox.showerror("Kesalahan Input", "Harap masukkan angka yang valid untuk semua input!")

# Fungsi untuk tombol evaluasi model
def on_evaluate_click():
    global best_k, best_accuracy, test_predictions
    # Menggunakan K terbaik yang telah ditemukan
    best_k, best_accuracy = find_best_k_kfold(train_features, train_labels, max_k=20, n_splits=5)
    
    # Menghitung prediksi pada data testing
    test_predictions = []
    for i in range(len(test_features)):
        prediction = knn_classify(train_features, train_labels, test_features[i], k=best_k)
        test_predictions.append(prediction)
    
    # Menampilkan akurasi
    accuracy = calculate_accuracy(train_features, train_labels, test_features, test_labels, best_k)
    accuracy_label.config(text=f"Akurasi: {accuracy:.2f}")
    
    # Menampilkan confusion matrix
    classes = np.unique(labels)
    plot_confusion_matrix(test_labels, test_predictions, classes)

# Load data
filename = 'data iris.csv'
features, labels = load_data(filename)

# Membagi data menjadi training dan testing set secara random
train_features, train_labels, test_features, test_labels = split_data(features, labels)

# Inisialisasi nilai K terbaik
best_k = 5
best_accuracy = 0.0
test_predictions = []

# Membuat GUI dengan Tkinter
root = tk.Tk()
root.title("Prediksi Kelas Iris")
root.geometry("300x400")

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

# Label Hasil Prediksi
result_label = tk.Label(root, text="Hasil: ", font=("Arial", 12))
result_label.grid(row=6, column=0, columnspan=2, pady=10)

# Tombol Evaluasi Model
evaluate_button = tk.Button(root, text="Evaluasi Model", command=on_evaluate_click)
evaluate_button.grid(row=7, column=0, columnspan=2, pady=10)

# Label Akurasi
accuracy_label = tk.Label(root, text="Akurasi: ", font=("Arial", 12))
accuracy_label.grid(row=8, column=0, columnspan=2, pady=10)

# Menjalankan aplikasi
root.mainloop()