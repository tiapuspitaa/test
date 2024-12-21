import pandas as pd
import streamlit as st

# Load Dataset
@st.cache_data
def load_dataset():
    dataset_path = 'processed_movies.csv'
    df = pd.read_csv(dataset_path)
    df['genres'] = df['genres'].fillna('').apply(lambda x: x.split(';'))
    return df

df = load_dataset()

# Jaccard Similarity Function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Recommend movies (on-demand computation)
def recommend_movies_by_title(movie_title, top_n=5):
    try:
        target_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return None  # Movie not found

    scores = []
    target_genres = set(df.loc[target_index, 'genres'])

    for i in range(len(df)):
        if i != target_index:
            other_genres = set(df.loc[i, 'genres'])
            similarity = jaccard_similarity(target_genres, other_genres)
            scores.append((i, similarity))

    # Sort by similarity and return top N
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [(x[0], x[1]) for x in scores[:top_n]]
    result = pd.DataFrame([
        {
            'title': df.loc[index, 'title'],
            'genres': '; '.join(df.loc[index, 'genres']),
            'similarity (%)': f"{similarity * 100:.2f}%"  # Convert to percentage
        }
        for index, similarity in top_indices
    ])
    return result

# Streamlit Interface
st.title("Movie Recommendation System")

# Dropdown to select movie
movie_title = st.selectbox("Select a Movie Title", df['title'].unique())

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies_by_title(movie_title, top_n=5)
    if recommendations is None:
        st.error("Movie not found. Please try again.")
    else:
        st.write("Recommended Movies with Similarity:")
        st.dataframe(recommendations)
