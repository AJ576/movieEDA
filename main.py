import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.set_page_config(layout="wide")

st.title("🎬 Movie Ratings Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('data/movie_ratings.csv')
    df_unexploded = pd.read_csv('data/movie_ratings_EC.csv')
    return df, df_unexploded

df, df_unexploded = load_data()

# Sidebar for genre selection
st.sidebar.header("Genre Analysis Options")


myGenres = sorted(df['genres'].unique())
selected_genre = st.sidebar.selectbox("Select one genre for detailed age analysis", myGenres)


selected_comparison_genres = st.sidebar.multiselect("Select genres to compare by age", myGenres, default=['Action', 'Drama', 'War', 'Sci-Fi'])

# Function to plot a horizontal bar chart
def plot_horizontal_bar(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    data.sort_values().plot(kind="barh", ax=ax, color='skyblue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)

# Function to plot a vertical bar chart
def plot_vertical_bar(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    data.sort_values(ascending=False).plot(kind="bar", ax=ax, color='lightgreen')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)

# Main content layout
col1, col2 = st.columns(2)

with col1:
    st.header("Genre Breakdown")
    genre_counts = df['genres'].value_counts()
    plot_horizontal_bar(genre_counts, "Breakdown of Genres", "Number of Movies", "Genre")

with col2:
    st.header("Average Rating per Genre")
    avg_rating_per_genre = df.groupby('genres')['rating'].mean()
    plot_vertical_bar(avg_rating_per_genre.sort_values(ascending=False), "Average Ratings Across Genres", "Genre", "Rating")

st.markdown("---")

st.header("Genre Popularity vs. Viewer Satisfaction")
genre_stats = df.groupby('genres')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'Average Rating', 'count': 'Number of Ratings'})
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(genre_stats['Number of Ratings'], genre_stats['Average Rating'], alpha=0.6, color='darkred')
for genre, row in genre_stats.iterrows():
    ax.annotate(genre, (row['Number of Ratings'], row['Average Rating']), fontsize=8, alpha=0.7)
ax.set_xlabel("Number of Ratings (Total Movies)")
ax.set_ylabel("Average Rating")
ax.set_title("Number of Ratings vs. Average Rating per Genre")
st.pyplot(fig)

st.markdown("---")

st.header("Average Rating by Release Year")
avg_rating_per_year = df.groupby('year')['rating'].mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(avg_rating_per_year.index, avg_rating_per_year.values, alpha=0.6, color='coral', label='Average Rating')


z = np.polyfit(avg_rating_per_year.index, avg_rating_per_year.values, 1)
p = np.poly1d(z)
ax.plot(avg_rating_per_year.index, p(avg_rating_per_year.index), "r--", label="Trend Line")

ax.set_xlabel("Release Year")
ax.set_ylabel("Average Rating")
ax.set_title("Average Rating by Release Year with Trend Line")
ax.legend()
st.pyplot(fig)

st.markdown("---")

st.header("Top-Rated Movies")
movie_stats = df_unexploded.groupby("title")["rating"].agg(["mean", "count"]).rename(columns={'mean': 'Average Rating', 'count': 'Number of Ratings'})

col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 5 with at least 50 ratings")
    top50 = movie_stats[movie_stats["Number of Ratings"] >= 50].sort_values("Average Rating", ascending=False).head(5)
    # 2. FIX: Display as a table
    st.table(top50)

with col4:
    st.subheader("Top 5 with at least 150 ratings")
    top150 = movie_stats[movie_stats["Number of Ratings"] >= 150].sort_values("Average Rating", ascending=False).head(5)
    # 2. FIX: Display as a table
    st.table(top150)

st.markdown("---")

col5, col6 = st.columns(2)

with col5:
    st.header(f"Rating by Viewer Age for {selected_genre}")
    if selected_genre:
        df_genre_single = df[df['genres'] == selected_genre]
        df_grouped_single = df_genre_single.groupby('age')['rating'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_grouped_single, x='age', y='rating', marker='o', color='purple', ax=ax)
        ax.set_title(f"Average Rating by Viewer Age for {selected_genre}")
        ax.set_xlabel("Viewer Age")
        ax.set_ylabel("Average Rating")
        st.pyplot(fig)

with col6:
    st.header("Compare Genres by Age")
    if selected_comparison_genres:
        df_genre_multi = df[df['genres'].isin(selected_comparison_genres)]
        df_grouped_multi = df_genre_multi.groupby(['genres', 'age'])['rating'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_grouped_multi, x='age', y='rating', hue='genres', marker='o', ax=ax)
        ax.set_title("Average Rating by Viewer Age for Selected Genres")
        ax.set_xlabel("Viewer Age")
        ax.set_ylabel("Average Rating")
        ax.legend(title="Genre")
        st.pyplot(fig)
    else:
        st.info("Please select at least one genre from the sidebar to view this analysis.")