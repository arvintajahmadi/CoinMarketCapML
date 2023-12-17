# CoinMarketCapMLImplementation

In This project, we used the CoinMarketCap data gathered in the last project to answer three different sets of questions in a group work.
My main contribution to the project was done in the third part.

This project was done in 2 weeks by a group of 3 and it was scored 89.5/100 by the project owner's (Quera IT Consulting and Education) auditing team.

## First Set
In the first one we tried different clustering methods on the data such as K-means, Silhouette and dbscan. We also used different denoising methods for each one and reported the results in visualizations to the project owner. The best k possible for the respective data was found in the process.

## Second Set
In the second set of questions, we used dendograms and Agglomerative Clustering to implement a hierarchical clustering on the respective data and find different aspects of using different attributes to cluster the data.

## Third Set
In the third set of questions we were asked to implement an ML-Based algorithm to predict tomorrow's situation of the XMR token in going up or down price-wise. In the process we used different models such as KnnClassifier, DecisionTreeClassifier and AdaboostClassifier. We also used two methods for the predicting. In the first method,after the preprocessing and data cleaning steps, we only used our historical data to predict all of the 30 days that were wanted to be predicted as a whole. In other words, the company wanted us to predict the situtation of the coin in a 30 day period. In the first method we didn't use any of the period's data to predict later days. For example we didn't use the first three days' data to predict the fourth day's situation. In the second method called "Backtesting", for the 30 day period that needed being predicted, not only we used all of the historical data before the 30 days, but also we used every day before the day being predicted from the 30 day period. For instance, we used all of the historical data and the first, second and third days' data to predict the fourth day. 
Finally we were able to achieve about 70% F1-Score for the problem.

## Additional
We tried to predict Bitcoin's same situation using simple data from its Wikipedia page to help the XMR predicting problem.
 