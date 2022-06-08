
## Fraud Data Analysis

Yazar: Mustafa Eroğlu

Tarih: 8 Haziran 2022

<p>
<p>Bu projede Amerika'daki firmaların muhasabe verileri içeren fınal veri seti ile çalıştım.
KMeans,Pytorch ve VarianceThreshold kullanarak analiz yaptım.</p>
</p>

<p> Veri setini görelim</p>

```
fraud_data
```

![image](/img/frauddata.png)

<p>fraud_data null değer içerdiği temizliyoruz.</p>

```
cleaned_fraud_data = fraud_data.dropna(axis=1)
cleaned_fraud_data
```
![image](/img/cleanedfrauddata.png)

<p>Temizlediğimiz verinin saçılım grafiğini çizdirelim.</p>

```
plt.figure(figsize = (12, 9))
plt.scatter(cleaned_ap, cleaned_sale)
plt.xlabel('ap')
plt.ylabel('sale')
plt.title('Visualization of data')

```

![image](/img/rawdata.png)

<p>StandardScaler ve KMeans algoritmasını uyguyalım.</p>

<p>Verinin saçılım grafiğini çizdirelim.</p>

```
plt.figure(figsize=(10,8))
sns.scatterplot(cleaned_ap,cleaned_sale,hue=df_fraud_label['fraud-or-not'],palette=['g','b'])
plt.title('Fraud Analysis')
plt.show()
```
![image](/img/kmeansfraud.png)

<p>Silhouette skorunu hesaplayalım.</p>

```
from sklearn.metrics import silhouette_score
score = silhouette_score(df_cleaned,kmeans.labels_,metric='euclidean')

#Print the score

print('Silhoutte Score: %.3f'%score)
```

<p>Silhoutte Score: 0.918</p>

<p>Temizlediğimiz veriyi tekrardan gözden geçirelim.</p>







<p>Okuduğunuz için teşekkürler </p>


