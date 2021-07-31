import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler


sensus_df= pd.DataFrame(
        {'Tinggi' : [176, 189, 156, 164, 178, 168, 160, 172, 190],
         'JK' : ['Pria', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Wanita', 'Pria', 'Pria', 'Pria'],
         'Berat' : [66, 79, 47, 56, 80, 70, 59, 71, 88]
         })

sensus_df

plt.scatter(sensus_df.Tinggi, sensus_df.Berat)
plt.title('Sensus Penduduk')
plt.xlabel('Tinggi')
plt.ylabel('Berat')
plt.show()

X_train = np.array(sensus_df[['Tinggi', 'JK']]) #
y_train = np.array(sensus_df[['Berat']])

print(f'X_train:\n{X_train}\n')
print(f'y_train: {y_train}')

#mengubah nilai string menjadi nilai biner
X_train_transposed = np.transpose(X_train) #mengubah posisi baris menjadi kolom, posisi kolom menjadi baris
print(f'X_train:\n{X_train}\n') #mencetak nilai X_train
print(f'X_transposed:\n{X_train_transposed}') #mencetak nilai X_train_transpose

lb = LabelBinarizer() #mengubah nilai string menjadi nilai biner (mengubah jenis kelamin menjadi nilai biner)
JK_binarised = lb.fit_transform(X_train_transposed[1]) #mengubah nilai posisi lb menjadi transpose

print(f'JK: {X_train_transposed}\n') #mencetak nilai X_train_transpose
print(f'JK_binarised:\n{JK_binarised}') #mencetak nilai hasil konversi ke biner

JK_binarised = JK_binarised.flatten() #flatten = konversi multidimensi array menjadi single dimensi array
JK_binarised

X_train_transposed[1] = JK_binarised #mengganti nilai X_train_transpose indeks ke 1 menjdi nilai JK_binarised
X_train = X_train_transposed.transpose() #mengubah posisi menjadi transpose

print(f'X_transposed:\n{X_train_transposed}\n') #mencetak nilai
print(f'X_train: {X_train}')

#trainning model menggunakan KNN (KNearestNeighbors) =>KNeighborsRegressor
K = 3 #banyaknya Neighbors (tetangga) yang akan digunakan untuk melakukan prediksi
model = KNeighborsRegressor(n_neighbors=K) #membentuk objek model dengan nilai parameter berupa nilai K
model.fit(X_train, y_train) #melakukan proses trainning model

#classification task = classifier
#regression task = regressor

#melakukan prediksi berat badan
X_new = np.array([[168, 1]]) #menambah nilai baru untuk dilakukan prediksi
X_new

y_pred = model.predict(X_new) #memprediksi nilai baru berat badan
y_pred #akan menampilkan akurasi prediksi berat badan

X_test = np.array([[170, 0], [164, 1], [178, 1], [180, 0]]) #nilai testing set /feature
y_test = np.array([63, 68, 69, 76]) #sekumpulan nilai target

print (f'X_test:\n{X_test}\n')
print (f'y_test: {y_test}')
 
y_pred = model.predict(X_test) #memprediksi berat badan terhadap nilai feature
y_pred #akan menampilkan akurasi prediksi berat badan

#menggunakan metrics r square untuk melakukan evaluasi model scikit learn
r_squared = r2_score(y_test, y_pred) #memanggil matrics r2 dengan parameter y_test, y_pred
print(f'R-quared: {r_squared}')

#menggunakan metrics mean absolut error untuk evaluasi 1
MAE = mean_absolute_error(y_test, y_pred)
print(f'MAE: {MAE}')

#menggunakan metrics mean square error untuk evaluasi 2
MSE = mean_squared_error(y_test, y_pred) #semakin kecil nilai MSE maka program akan semakin baik
print(f'MSE: {MSE}')

#permasalahan scalling pada features
#dalam satuan milimeter
X_train = np.array ([[1680, 1], [1700, 0]]) #berisi sekumpulan nilai feature untuk trainning set
X_new = np.array([[1600, 1]]) #data yg mau di prediksi

[euclidean(X_new[0], d) for d in X_train]

#dalam satuan meter
X_train = np.array ([[1.68, 1], [1.7, 0]])
X_new = np.array([[1.6, 1]])

[euclidean(X_new[0], d) for d in X_train]

#menerapkan standard scaler guna mengatasi masalah ketidakcocokan distance terhadapan satuan pengukuran
ss = StandardScaler()

#dalam satuan milimeter
X_train = np.array ([[1680, 1], [1700, 0]]) #berisi sekumpulan nilai feature untuk trainning set
X_train_scaled = ss.fit_transform(X_train)
print(f'X_train_scaled:\n{X_train_scaled}\n')

X_new = np.array([[1600, 1]])
X_new_scaled = ss.transform(X_new)
print(f'X_new_scaled: {X_new_scaled}')

jarak = [euclidean(X_new_scaled[0], d) for d in X_train_scaled]
print (f'jarak: {jarak}')

#dalam satuan meter

X_train = np.array ([[1.68, 1], [1.7, 0]]) #berisi sekumpulan nilai feature untuk trainning set
X_train_scaled = ss.fit_transform(X_train)
print(f'X_train_scaled:\n{X_train_scaled}\n')

X_new = np.array([[1.6, 1]])
X_new_scaled = ss.transform(X_new)
print(f'X_new_scaled: {X_new_scaled}')

jarak = [euclidean(X_new_scaled[0], d) for d in X_train_scaled]
print (f'jarak: {jarak}')

#menerapkan features scaleing pada KNN
#training_set
X_train = np.array ([[189, 0], [162, 1], [171, 1], [180, 0], [166, 1], [192, 1], [168, 0], [155, 0], [174, 1]])
y_train = np.array([78, 61, 70, 77, 65, 86, 67, 50, 73])

#test set
X_test = np.array ([[178, 0], [164, 0], [178, 0], [172, 1]])
y_test = np.array ([77, 63, 77, 70])


#features scaling
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

print(f'X_train_scaled:\n{X_train_scaled}\n')
print(f'X_test_scaled:\n {X_test_scaled}\n')

#training dan evaluasi model
model.fit(X_train_scaled, y_train) #melakukan proses trainning model
y_pred = model.predict(X_test_scaled)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

print(f'MAE: {MAE}')
print(f'MSE: {MSE}')

