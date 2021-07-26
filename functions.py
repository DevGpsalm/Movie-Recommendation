from pandas.io.formats.format import GenericArrayFormatter


def ratings_ranked_df(data, vote_count = 1000):
        global ratings_ranked
        '''
        This function returns a dataframe rank of movies in descending order for movies with atleast 1000 votes

        INPUT
        data - the movies dataframe with the needed columns
        Vote_count: This is the minimum number of votes casted for which a movie will qualify as one to be recommended. This has 
        been set to a default value of 1000. 
    
        OUTPUT
        ratings_ranked - a dataframe with movies that are sorted by highest voting and popularity
        '''
        #Movies with the same ratings, will have the most popular movie will come first
        ratings_ranked = data.sort_values(['vote_average', 'popularity'], ascending = False)
        ratings_ranked = ratings_ranked[ratings_ranked.vote_count >= vote_count]
        
        return ratings_ranked


def popular_movies(ratings_ranked, n_top, genres=None):
    '''   
    INPUT:    
    ranked_movies: a pandas dataframe of the already ranked movies based on avg rating, popularity,
    n_top: the number of top movies you wish to generate
    genres: a list of strings, genres of movies
    
    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order of highest to lowest ratings
    '''
        
    if genres is not None:
        num_genre_match = ratings_ranked[genres].sum(axis=1)
        ratings_ranked = ratings_ranked.loc[num_genre_match > 0, :]
            
            
            
    # create top movies list 
    top_movies = list(ratings_ranked.original_title[:n_top])

    return top_movies


def extract_dict_names(df, column):
    '''Function that creates a DataFrame from a column of dictionary string
    INPUT - The DataFrame; and the column name whose values are to be extracted. The column should be in quote
    OUTPUT - DataFrame containing the original data merged with the new column containing just the values of 
    the preferred dict string
    ''' 
    import pandas as pd
    import ast
    cols_df = pd.DataFrame(["|".join([i.get('name') for i in ast.literal_eval(i)]) for i in df[column]], columns = [column])
    return cols_df


def movie_recommender (df,film_name, top_n = 10):
    '''
    Function for making recommendation based on the content similarity of movies for the movie provided
    INPUT: DataFrame as df, movie as film_name
    OUTPUT: Dataframe of recommended movies by popularity
    '''  
    from scipy.sparse import csr_matrix
    
    import pandas as pd

    import ast

    import sklearn.metrics.pairwise as pw
    from scipy import sparse
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.preprocessing import MinMaxScaler

    pivot_item_based = pd.pivot_table(df, index='title', values = ['vote_average', 'vote_count'])  
    sparse_pivot = sparse.csr_matrix(pivot_item_based)
    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
    recommender = pw.cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(recommender, columns=pivot_item_based.index, index=pivot_item_based.index)

    #Item Rating Based Cosine Similarity
    cosine_df = pd.DataFrame(recommender_df[film_name].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['title','cosine_sim']
    cosine_df = cosine_df.iloc[1: , :]
    cosine_df.cosine_sim.round(6)
    return cosine_df.title.head(top_n)