######################

# HYBRID RECOMMENDER #

######################

## GÖREV-1 ##

# Veri Ön İşleme #

import pandas as pd

pd.set_option("display.max_columns",None)

movie = pd.read_csv("movie.csv")

ratings = pd.read_csv("rating.csv")

df = movie.merge(ratings,on="movieId",how="left")

# Veriyi anlamak için gerekenleri yazdırır.
def dataframe_info(df):

    print("-----Head-----","\n",df.head())

    print("\n-----Tail-----","\n",df.tail())

    print("\n-----Shape-----","\n",df.shape)

    print("\n-----Columns-----","\n",df.columns)

    print("\n-----Index-----","\n",df.index)

    print("\n-----Statistical Values-----","\n",df.describe().T)

dataframe_info(df)


# 1000 üzeri yorum yapılan filmlerin seçilmesi:

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

user_movie_df = common_movies.pivot_table(index = ['userId'],
                                          columns = ['title'],
                                          values = 'rating')


user_movie_df.shape

## GÖREV-2 ##

# Öneri yapılacak kullanıcının izlediği filmleri belirleme #

# Random user seçme #
random_user = int(pd.Series(user_movie_df.index).sample(1,random_state=7).values)

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)

## GÖREV-3 ##

# Aynı filmleri izleyen diğer kullanıcıların verisine ve ID'lerine erişme #

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]
user_movie_count
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count['movie_count']>perc]['userId']
users_same_movies

## GÖREV-4 ##

# Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleme #

# 1.adım -> User ve diğer user'ların verilerini bir araya getireceğiz.
# 2.adım -> Korelasyon df'ni oluşturacağız.
# 3.adım -> En benzer kullanıcıları bulacağız.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df,columns =["corr"])

corr_df.index.names = ["user_id_1","user_id_2"]

corr_df.head()

corr_df = corr_df.reset_index()

corr_df.head()

top_users = corr_df[(corr_df["user_id_1"]==random_user) & (corr_df["corr"]>=0.6)][['user_id_2',"corr"]].reset_index(drop=True)

top_users.rename(columns = {"user_id_2":"userId"},inplace = True)

top_users_ratings = top_users.merge(ratings[['userId',"movieId","rating"]],how="inner")

top_users_ratings= top_users_ratings[top_users_ratings["userId"]!= random_user]

## GÖREV-5 ##

# Weighted Average Recommendation Score'u hesaplama ve ilk 5 filmi tutma #

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating':"mean"})

recommendation_df = recommendation_df.reset_index()

movies_recommendation = recommendation_df[recommendation_df['weighted_rating'] > 3.3].sort_values("weighted_rating",ascending =False)

movies_recommendation

user_based_rec = movies_recommendation.merge(movie[["movieId", "title"]]).head()

print("\n---------------- User-Based 5 Film Önerisi ----------------\n")
user_based_rec['title']

## GÖREV-6 ##

# Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based recommendation #

movieId= ratings[(ratings['userId'] == random_user) & (ratings["rating"] == 5)].\
           sort_values(by='timestamp',ascending = False)["movieId"][0:1].values[0]


movie_name = movie.iloc[movieId]

movie_name

item_based_rec = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head()

item_based_rec = item_based_rec.reset_index()

item_based_rec = item_based_rec.iloc[:,:1]


hybrid_recommendation = pd.concat([user_based_rec["title"], item_based_rec["title"]], ignore_index=True)
hybrid_recommendation