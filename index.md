
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

```
cleaned_fraud_data.head()
```

![image](/img/frauddatahead.png)

```
cleaned_fraud_data.shape
```
<p>(146045, 35)</p>

<p>VarianceThreshold algoritmasıyla düşük varyanslı featureları silelim.</p>

<p>Bu işlem sonucu np array döner. var_thres_clean değişkenine atayalım.</p>

```
var_thres_clean = sel.fit_transform(X)
var_thres_clean
```
![image](/img/varthresclean.png)

<p>Bu işlemden sonra kalan featureların indislerine bakalım.</p>

```
features = sel.get_support(indices=True)
features
```
![image](/img/unremovedfeatures.png)

<p>var_thres_clean'i DataFrame'e dönüştürelim.</p>

```
variance_cleaned = pd.DataFrame(data=var_thres_clean,columns=cleaned_fraud_data.iloc[:,features].columns)
variance_cleaned.head()
```
![image](/img/variancecleaned.png)

<p>KMeans algoritmasını uygulayıp ve label verdikten sonra saçılım grafiğini çizdirelim.</p>


```
plt.figure(figsize=(10,8))
sns.scatterplot(variance_cleaned_labeled_ap,variance_cleaned_labeled_sale,hue=variance_cleaned_labeled['fraud-or-not'],palette=['g','b'])
plt.title('Fraud Analysis')
plt.show()
```
![image](/img/variancecleanedkmeans.png)

<p>Silhoutte skorunu hesaplayalım.</p


```
#silhouette_score 
```
 
``` 
from sklearn.metrics import silhouette_score 
score = silhouette_score(var_thres_clean,kmeans.labels_,metric='euclidean')

#Print the score

print('Silhoutte Score: %.3f'%score)
```
 
 
<p>Silhoutte Score: 0.750</p>

<p>PyTorch'u kuralım.</p>

```
import torch
import numpy as np

!pip install kmeans_pytorch
```
![image](/img/pytorchkurulum.png)

<p>PyTorch KMeans kullanımı: </p>

```
from kmeans_pytorch import kmeans
torch_var_thres_clean = torch.from_numpy(var_thres_clean)
cluster_ids_x, cluster_centers = kmeans(
    X=torch_var_thres_clean, num_clusters=2, distance='euclidean'
)
```

<p>running k-means on cpu..</p>
<p>[running kmeans]: 3it [00:00,  6.36it/s, center_shift=0.000000, iteration=3, tol=0.000100]</p>


<p>Silhoutte skoru hesaplayalım.</p>

```
from sklearn.metrics import silhouette_score
score = silhouette_score(var_thres_clean,cluster_ids_x,metric='euclidean')

#Print the score

print('Silhoutte Score: %.3f'%score)
```


<p>Silhoutte Score: 0.750</p>





<p>Okuduğunuz için teşekkürler </p>


