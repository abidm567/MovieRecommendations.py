#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The aim is to develop a movie recommendation system where similar movies are recommended by analysing the ratings of previous users who
#have watched that movie ,rated that movie and given similar ratings to other movies,matching the similarities with some quality score.


import sys
from math import sqrt
from pyspark import SparkContext  #Importing Spark Context. Its like entry point of Spark cluster on which we can do the analysis by giving some functions.

# Using the existing SparkContext
sc = SparkContext.getOrCreate()

def loadMovieNames(): #Defining function loadMovieNames
    movieNames = {}
    with open("/dbfs/FileStore/tables/ml_100k/u.item", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

def makePairs(user_ratings): #Defining function makepairs with Users and movie,ratings as key-value pairs
    (movie1, rating1) = user_ratings[0]
    (movie2, rating2) = user_ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates(user_ratings):  #Removing duplicates where movie1 might be same as movie2
    (movie1, rating1) = user_ratings[0]
    (movie2, rating2) = user_ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs): # Computing similarities between set of ratingPairs
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if denominator:
        score = (numerator / float(denominator))

    return (score, numPairs)

print("\nLoading movie names...")
nameDict = loadMovieNames() #Mapping the input movie name to dataframe we created earlier from loadMovieNames function

data = sc.textFile("dbfs:/FileStore/tables/ml_100k/u.data") #Reading data by Spark context

ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))  #Mapping the columns and data we want to the ratings dataframe

joinedRatings = ratings.join(ratings)

uniqueJoinedRatings = joinedRatings.filter(filterDuplicates) #Creating Dataframe out of filtered out duplicates from filterDuplicates function

#  Map to movie1,movie2 pairs
moviePairs = uniqueJoinedRatings.map(makePairs)

#Grouping all the movie1,movie2 pairs
moviePairRatings = moviePairs.groupByKey()
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()

# If a movie ID is provided as input, look up the top 10 similar movies
if len(sys.argv) > 1:
    scoreThreshold = 0.10 #Setting similarity as 0.10
    coOccurenceThreshold = 50 

    movieID = int(sys.argv[1]) #taking first argument which is input of movie id

    filteredResults = moviePairSimilarities.filter(lambda pair_sim: \ #filtering the movie rating pairs where the given movie ID is there
        (pair_sim[0][0] == movieID or pair_sim[0][1] == movieID) and \
        pair_sim[1][0] > scoreThreshold and pair_sim[1][1] > coOccurenceThreshold) #Setting conditions so that our desired threshold values are passed

    results = filteredResults.map(lambda pair_sim: (pair_sim[1], pair_sim[0])).sortByKey(ascending=False).take(10) #Sorting the similar movies by scores and taking top10 movies.

    print(f"Top 10 similar movies for {nameDict[movieID]}:") #Printing the Top 10 similar movies to the given movie ID
    for result in results:
        (sim, pair) = result
        similarMovieID = pair[0]
        if similarMovieID == movieID:
            similarMovieID = pair[1]
        print(f"{nameDict[similarMovieID]}\tscore: {sim[0]}\tstrength: {sim[1]}")

