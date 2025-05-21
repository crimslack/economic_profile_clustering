#gerekli kütüphaneleri ekledik
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# CSV dosyasını oku
df = pd.read_csv("Country-data.csv")

df['country'] = df['country'].str.strip()

#country sütununu çıkarttık
X = df.drop(columns=["country"])

# Verileri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ölçeklendirilmiş veriyi DataFrame'e dönüştür (sütun isimlerini koru)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# İlk 5 satırı yazdır
# print(X_scaled_df.head())

# DBSCAN modelini oluştur
# eps: komşuluk yarıçapı (parametre, deneyerek ayarla)
# min_samples: bir kümenin oluşması için gereken minimum nokta sayısı

dbscan = DBSCAN(eps=1.5, min_samples=5)

# Modeli veriye uygula ve küme etiketlerini al
clusters = dbscan.fit_predict(X_scaled)

# Orijinal dataframe'e küme etiketlerini ekle
df['cluster'] = clusters

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DataFrame oluştur (görselleştirme için)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = df['cluster']
pca_df['country'] = df['country']

# Grafik çizimi
plt.figure(figsize=(10,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='tab10', s=100, legend='full')
plt.title("DBSCAN Kümeleme Sonuçları (PCA İle 2 Boyuta İndirilmiş)")
plt.xlabel("Birinci Bileşen (PC1)")
plt.ylabel("İkinci Bileşen (PC2)")
plt.legend(title='Küme')
plt.grid(True)
plt.show()


# Küme etiketlerini ve veri türlerini yazdır
print(df.columns)
print(df['cluster'].value_counts())
print(df.dtypes)

cluster_profiles = df.groupby('cluster').mean(numeric_only=True)
print(cluster_profiles)

print(df['cluster'].value_counts())

# Küme sayısını ve her kümenin profilini yazdır
print("Küme Sayısı:")
print(df['cluster'].value_counts())
print("\nKüme Profilleri:")
print(cluster_profiles)
input("\nKüme profilleri oluşturulmuştur. Devam etmek için Enter'a basın...")


# Gürültü noktalarını yazdır
print("\nGürültü Noktaları:")
print(df[df['cluster'] == -1][['country']])
df.to_csv("clustered_countries.csv", index=False)

