import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the dataset
df = pd.read_csv('idmanual.tsv', sep='\t')
print(df.head())

# Get rid of rows where Class is A or B or 200 or 000
df = df[df['Class'] != 'A']
df = df[df['Class'] != 'B']
df = df[df['Class'] != '200']
df = df[df['Class'] != '000']


# Reset the index
df.reset_index(drop=True, inplace=True)

# Get the embeddings for the descriptions
embeddings = model.encode(df['Description'], show_progress_bar=True)

# Add embeddings to the dataframe
df['Embeddings'] = embeddings.tolist()

print(df.head())

# Sort dataframe by class
df = df.sort_values(by='Class')
# Reset the index
df.reset_index(drop=True, inplace=True)

print(df.head())
print(df.Class.unique())
#print 46th row
print(df.iloc[46])
labels = []
reduced_embeddings = []
silhouette_score_total = 0
# Plot each class in a different plot
for i in df.Class.unique():
    df_class = df[df['Class'] == i]
    # Reduce the dimensionality of the embeddings to 2 using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(np.array(df_class['Embeddings'].to_list()))
    # Save the reduced embeddings
    reduced_embeddings.extend(embeddings_tsne.tolist())
    #find the number of embeddings in the class
    n_embeddings = len(embeddings_tsne)
    #divide by 2 and take the square root to get the number of clusters
    n_clusters = int(np.sqrt(n_embeddings/2))
    kmeans = KMeans(n_clusters, random_state=42).fit(embeddings_tsne)
    # Save the cluster labels
    labels.extend(kmeans.labels_.tolist())
     #calculate the silhouette score
    silhouette_avg = silhouette_score(embeddings_tsne, kmeans.labels_)
    #add the silhouette score to the total
    silhouette_score_total += silhouette_avg
# Calculate the average silhouette score
silhouette_score_total = silhouette_score_total / len(df.Class.unique())

print(len(reduced_embeddings))
# Create a new dataframe with the class, reduced embeddings, and cluster labels
df_new = pd.DataFrame()
df_new['Class'] = df['Class']
df_new['Description'] = df['Description']
df_new['Embeddings'] = reduced_embeddings
df_new['Cluster'] = labels


print(df_new.head())
print(silhouette_score_total)

#save new dataframe to a tsv file
df_new.to_csv('idmanual_clusters.tsv', sep='\t', index=False)


