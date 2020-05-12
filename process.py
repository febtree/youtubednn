import conf
import pandas as pd
import numpy as np


def genreNum(genres):
    return len(genres.split("|"))


# read dataframes
ratingDtypes = {"uid": np.int16, "mid": np.int16, "ratings": np.float, "timestamp": np.int64}
ratings = pd.read_csv(conf.DATA_DIR + "ratings.dat",
                      sep="::", header=None,
                      names=["uid", "mid", "rating", "timestamp"],
                      dtype=ratingDtypes,
                      engine='python')
print("ratings shape:", ratings.shape)

userDtypes = {"uid": np.int16, "age": np.int16, "occupation": np.int16}
users = pd.read_csv(conf.DATA_DIR + "users.dat",
                    sep="::", header=None,
                    names=["uid", "gender", "age", "occupation", "zipcode"],
                    dtype=userDtypes,
                    engine='python')
print("users shape:", users.shape)

movieDtypes = {"mid": np.int16}
movies = pd.read_csv(conf.DATA_DIR + "movies.dat",
                     sep="::", header=None,
                     names=["mid", "title", "genres"],
                     dtype=movieDtypes,
                     engine='python')
print("movies shape:", movies.shape)

# join dataframs to get training data
trainingSet = ratings.merge(users, how='left', on='uid').merge(movies, how='left', on='mid')
trainingSet["genreNum"] = trainingSet["genres"].apply(genreNum)
maxGenreNum = trainingSet["genreNum"].max()
genreCols = ["gen"+str(i) for i in range(maxGenreNum)]
trainingSet[genreCols] = trainingSet.genres.str.split("|", expand=True)
features = genreCols + ["uid", "mid", "gender", "age", "occupation", "rating"]
trainingSet = trainingSet[features]
print(trainingSet.head(10))

# encode to get samples
genres = pd.unique(trainingSet[genreCols].values.ravel())
print(genres)
