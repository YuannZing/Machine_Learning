import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Membaca data dari file CSV menggunakan pandas
def load_data(filename):
    data = pd.read_csv(filename, delimiter=';')  # Ganti dengan delimiter yang sesuai
    features = data.iloc[:, :-1].values  # Semua kolom kecuali yang terakhir sebagai fitur
    labels = data.iloc[:, -1].values     # Kolom terakhir sebagai label
    return features, labels

# Membaca data
filename = 'data iris 1.csv'  # Ganti dengan nama file CSV Anda
features, labels = load_data(filename)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Menyimpan akurasi untuk berbagai nilai K
k_range = range(1, 22)  # Mencoba nilai K dari 1 sampai 21
accuracies = []

# Menguji KNN untuk berbagai nilai K
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Menggunakan cross-validation untuk mendapatkan akurasi
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    
    # Menyimpan rata-rata akurasi untuk K yang diuji
    accuracies.append(np.mean(cv_scores))

# Menampilkan grafik
plt.figure(figsize=(10,6))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='b')
plt.title('Akurasi KNN untuk Nilai K yang Berbeda', fontsize=14)
plt.xlabel('Nilai K', fontsize=12)
plt.ylabel('Akurasi Rata-rata', fontsize=12)
plt.xticks(k_range)  # Menampilkan semua nilai K pada sumbu X
plt.grid(True)
plt.show()

# Mencari nilai K terbaik berdasarkan akurasi tertinggi
best_k = k_range[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print(f"Nilai K terbaik adalah {best_k} dengan akurasi rata-rata {best_accuracy:.4f}")
