from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset
movies = pd.read_csv('movies.csv')  # Make sure your dataset is in the same directory

movies['tags']=movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['tagline'] + ' ' + movies['cast'] + ' ' + movies['director']

df=movies[['id','title','genres','keywords','tagline','cast','director','tags']]

# Preprocess the data
cv = CountVectorizer(max_features=1000, stop_words='english')
vec = cv.fit_transform(df['tags'].values.astype('U')).toarray()
sim = cosine_similarity(vec)

# Recommendation function
def recommend(movie_title):
    index = df[df['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(sim[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [df.iloc[i[0]].title for i in distances[1:6]]
    return recommended_movies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie = request.form['movie']
    recommendations = recommend(movie)
    return render_template('index.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
