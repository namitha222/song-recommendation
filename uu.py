import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load data
songs_df = pd.read_csv('songs.csv')

# Preprocess data
songs_df.drop_duplicates(inplace=True)
songs_df.fillna(0, inplace=True)
songs_df['genre'] = pd.factorize(songs_df['genre'])[0]

# Feature engineering
X = songs_df[['tempo', 'key', 'duration', 'genre']]

# Train the model
knn = NearestNeighbors(n_neighbors=10)
knn.fit(X)

# Get recommendations
song_idx = 5 # Example song index
distances, indices = knn.kneighbors(X.iloc[song_idx, :].values.reshape(1, -1))
recommended_songs = songs_df.iloc[indices[0][1:], :]
