from flask import Flask,render_template,request
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

movies= pd.read_csv('tmdb_3000_movies.csv')
df = pd.read_csv('tmdb_3000_credits.csv')
df.columns = ['id', 'tittle', 'cast', 'crew']
movies = movies.merge(df, on='id')
mov=pd.read_csv('tmdb_3000_movies.csv')
j=mov[['original_title']]

tfidf = TfidfVectorizer(analyzer = 'word',stop_words = 'english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

cosin_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
index_of_movies = pd.Series(movies.index,index=movies['title']).drop_duplicates()
def get_recommendations(title, cosin_sim=cosin_sim):
    idx = index_of_movies[title]
    sim_scores = list(enumerate(cosin_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True) 
    sim_scores = sim_scores[1:31]
    movies_idx = [i[0] for i in sim_scores]
    return movies['title'].iloc[movies_idx]
    
features = ['cast', 'crew', 'keywords', 'genres']
for f in features:
    movies[f] = movies[f].apply(literal_eval)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [ i['name'] for i in x]
        if len(names)  > 3:
            names = names[:3]
        return names
    return []
#apply all functions
movies['director'] = movies['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for f in features:
    movies[f] = movies[f].apply(get_list)
#striping
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''
features = ['cast', 'keywords', 'director', 'genres']
for f in features:
    movies[f] = movies[f].apply(clean_data)
#creating a SOUP
def create_soup(x):
    return ' '.join(x['keywords'])+' '+' '.join(x['cast'])+' '+x['director']+' '+' '.join(x['genres'])
movies['soup'] = movies.apply(create_soup, axis=1)
#count Vectorizer

count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(movies['soup'])
# finding similarity matrix

cosin_sim2 = cosine_similarity(count_matrix, count_matrix)

reader = Reader()
ratings = pd.read_csv('ratings_small.csv')
data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
svd=SVD()
cross=cross_validate(svd,data,measures=['RMSE'],cv=3,verbose=True)
train = data.build_full_trainset()
sfit=svd.fit(train)
movie_id = pd.read_csv('links.csv')[['movieId', 'tmdbId']]
movie_id.columns = ['movieId','id']
movie_id = movie_id.merge(movies[['title', 'id']], on='id').set_index('title')
index_map = movie_id.set_index('id')
def recommend_for(userId,title):
    l=[]
    try:
        index = index_of_movies[title]
    except:
        return l
    tmdbId = movie_id.loc[title]['id']
    

    #content based
    sim_scores = list(enumerate(cosin_sim2[int(index)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:30]
    movie_indices = [i[0] for i in sim_scores]
    mv = movies.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
    mv = mv[mv['id'].isin(movie_id['id'])]
    # CF
    mv['est'] = mv['id'].apply(lambda x: svd.predict(userId, index_map.loc[x]['movieId']).est)
    #mv = mv.sort_values('est', ascending=False)
    k=mv.head(5)
    l=k.index.tolist()
    return l
app=Flask(__name__)
@app.route('/')
def recom():
    return render_template('home.html')
@app.route('/recommend',methods=['POST'])
def recommend():
    data1=request.form['a']
    u=1
    n=data1.capitalize()
    genre=["Fantasy","Action","Adventure","Science Fiction","Crime","Drama","Thriller","Animation","Family","Western","Romance","Comedy","Horror","Mystery","War","History","Music",]
    if len(n)>=3:
        if n in genre:
            v=mov["genres"]
            cou=2
            l=[]
            for i in v:
                i=str(i)
                if n in i:
                    l.append(cou)
                if len(l)==5:
                    break
                cou+=1
            POSTER_FOLDER=os.path.join('static','posters')
            app.config['UPLOAD_FOLDER']=POSTER_FOLDER
    
            if len(l)!=0:
                a=str(l[0]+2)+".jpg"
                b=str(l[1]+2)+".jpg"
                c=str(l[2]+2)+".jpg"
                d=str(l[3]+2)+".jpg"
                e=str(l[4]+2)+".jpg"
                full_filename1=os.path.join(app.config['UPLOAD_FOLDER'],a)
                full_filename2=os.path.join(app.config['UPLOAD_FOLDER'],b)
                full_filename3=os.path.join(app.config['UPLOAD_FOLDER'],c)
                full_filename4=os.path.join(app.config['UPLOAD_FOLDER'],d)
                full_filename5=os.path.join(app.config['UPLOAD_FOLDER'],e)
                return render_template("after.html",user_image1=full_filename1,user_image2=full_filename2,user_image3=full_filename3,user_image4=full_filename4,user_image5=full_filename5,user_image6=n)
            else:
                return render_template("error.html")
        else:
        
            m=j[j['original_title'].str.contains(n)]
            try:
                m=m["original_title"].values[0]
            except:
                return render_template("error.html")
            l=recommend_for(u,m)
            # to display images in new web page
            POSTER_FOLDER=os.path.join('static','posters')
            app.config['UPLOAD_FOLDER']=POSTER_FOLDER
    
            if len(l)!=0:
                a=str(l[0]+2)+".jpg"
                b=str(l[1]+2)+".jpg"
                c=str(l[2]+2)+".jpg"
                d=str(l[3]+2)+".jpg"
                e=str(l[4]+2)+".jpg"
                full_filename1=os.path.join(app.config['UPLOAD_FOLDER'],a)
                full_filename2=os.path.join(app.config['UPLOAD_FOLDER'],b)
                full_filename3=os.path.join(app.config['UPLOAD_FOLDER'],c)
                full_filename4=os.path.join(app.config['UPLOAD_FOLDER'],d)
                full_filename5=os.path.join(app.config['UPLOAD_FOLDER'],e)
                return render_template("after.html",user_image1=full_filename1,user_image2=full_filename2,user_image3=full_filename3,user_image4=full_filename4,user_image5=full_filename5,user_image6=m)
            else:
                return render_template("error.html")
    else:
        return render_template("error.html")

if __name__=='__main__':
    app.run(debug =False)