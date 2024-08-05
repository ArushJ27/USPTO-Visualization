import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the dataset
df = pd.read_csv('idmanual.tsv', sep='\t')
print(df.head())

# Get rid of rows where Class is A or B or 200
df = df[df['Class'] != 'A']
df = df[df['Class'] != 'B']
df = df[df['Class'] != '200']


# Reset the index
df.reset_index(drop=True, inplace=True)

# Compute the embeddings for testing
embeddings = model.encode(df.Description[:10000], show_progress_bar=True)



#create a new data frame with the class and the 
df_pca = pd.DataFrame()
df_pca['Embeddings'] = list(embeddings)
df_pca['Class'] = df['Class'[:10000]]

df_pca = df_pca.sort_values(by='Class')
print(df_pca.Class.unique())
print(df_pca.head())

labels = []
reduced_embeddings = []

#plot each class in a different plot
for i in df_pca.Class.unique():
    df_class = df_pca[df_pca['Class'] == i]
    # Reduce the dimensionality of the embeddings to 10 using PCA
    pca = PCA(n_components=10)
    embeddings_pca = pca.fit_transform(df_class['Embeddings'].to_list())
    reduced_embeddings.append(embeddings_pca.tolist())
    #cluster the embeddings
    kmeans = KMeans(n_clusters=8, random_state=42).fit(embeddings_pca)
    #appened the labels to the list
    labels.append(kmeans.labels_)
    #plot the clusters
    plt.scatter((embeddings_pca)[:, 0],(embeddings_pca)[:, 1], c=kmeans.labels_)
    plt.title(i)
    plt.show()

#add labels to the data frame
df_pca['Cluster Labels'] = np.concatenate(labels)

#replace df_pcs Embeddings column with the new embeddings
df_pca['Embeddings_pca'] = np.concatenate(reduced_embeddings)

#print first 15 rows
print(df_pca.head(15)) 



























