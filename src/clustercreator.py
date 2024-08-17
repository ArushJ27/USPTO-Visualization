import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json

# Read the JSON file
with open('uspto_classes.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the JSON data to a DataFrame
df_json = pd.DataFrame(list(data.items()), columns=['class_id', 'class_name'])
print(df_json.head())

nltk.download('stopwords')
stemmer = PorterStemmer()

# Read new file
df = pd.read_csv('idmanual_clusters.tsv', sep='\t')
print(df.head())

# Convert embeddings from strings to lists of floats
df['Embeddings'] = df['Embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Sort dataframe by class and each class by cluster
df = df.sort_values(by=['Class', 'Cluster'])
df.reset_index(drop=True, inplace=True)
print(df.head())

# Save the common words in a new column
df['Common Words'] = ''

# Plot the embeddings of each class in a different plot
for i in df['Class'].unique():
    df_class = df[df['Class'] == i].copy()
    embeddings = np.stack(df_class['Embeddings'].to_list())
    
    # Remove stopwords from descriptions
    stop_words = set(stopwords.words('english'))
    df_class['Description'] = df_class['Description'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    
    # Plot the embeddings using color as the cluster number
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=df_class['Cluster'], label=i)
    
    # For each cluster, label it with the 2 most common words in the descriptions
    for j in df_class['Cluster'].unique():
        df_cluster = df_class[df_class['Cluster'] == j]
        words = ' '.join(df_cluster['Description']).split()
        
        # All words to lowercase
        words = [word.lower() for word in words]

        #create a dictionary to store the frequency of each word and the word it gets stemmed to
        word_freq = {}
        for word in words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word in word_freq:
                word_freq[stemmed_word].add(word)
            else:
                word_freq[stemmed_word] = {word}
       
       # Remove stopwords
        words = [word for word in words if word not in stop_words]
        common_words = pd.Series(words).value_counts().index.tolist()[:2]

        #for the common words, find the word that has the highest frequency
        for word in common_words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word in word_freq:
                common_words[common_words.index(word)] = max(word_freq[stemmed_word], key=lambda x: len(x))

        # Save the common words to a new column in the original dataframe
        df.loc[df_cluster.index, 'Common Words'] = ' '.join(common_words)
        
        # Find the centroid of the cluster
        centroid = np.mean(df_cluster['Embeddings'].to_list(), axis=0)
        plt.text(centroid[0], centroid[1], ' '.join(common_words))
    
    plt.title(i)
    plt.show()

print(df.head())

# Create a dictionary for class_id to class_name mapping
class_mapping = df_json.set_index('class_id')['class_name'].to_dict()
print(class_mapping)

df['Class'] = df['Class'].astype(str)

#use the dictionary to replace df.class with class names
df['Class'] = df['Class'].map(class_mapping)
print(df.head())

#Remove stopwords from descriptions and remove the column
df['Description'] = df['Description'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


# Save the dataframe to a new TSV file
df.to_csv('idmanual_clusters_common_words.tsv', sep='\t', index=False)
print(df.head())
