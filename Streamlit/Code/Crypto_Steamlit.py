import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import seaborn as ns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score,silhouette_samples
import seaborn as sns
from PIL import Image
from sklearn.cluster import DBSCAN
import time
from sklearn.preprocessing import OneHotEncoder
import mwclient




cf = pd.read_csv(r"cf.csv")

st.title("Cryptocurrency Analysis")

report_choice = st.sidebar.selectbox("Select a Report", ["Report 1", "Report 2", "Report 3"])


if report_choice == "Report 1":
    st.header("Report 1")

    data = pd.read_csv(r"Q1_data.csv")
    st.subheader("First lets take a look at the data :")
    data

    # code :

    data = pd.read_csv(r"Q1_data.csv")
    
    from sklearn.preprocessing import StandardScaler
    X=data.copy()
    X['market_cap']=StandardScaler().fit_transform(np.array(X['market_cap']).reshape(-1, 1))
    X['volume']=StandardScaler().fit_transform(np.array(X['volume']).reshape(-1, 1))
    features = data[['market_cap','volume']]

    colors = {'BTC': 'blue', 'BNB': 'green', 'ETH': 'red', 'USDT': 'purple'}
    alphas= {'BTC': 'Blues', 'BNB':'YlGn', 'ETH':'Reds', 'USDT':'Purples'}
    np.random.seed(42)
    alpaa_rand=np.random.rand(364)
    plt.figure(figsize=(24, 12))
    for coin in colors:
        subset = data[data['symbol'] == coin]
        plt.scatter(subset['market_cap'], subset['volume'],s=100, label=coin, c=alpaa_rand,cmap=alphas[coin],alpha=0.65)

    plt.title('Coin Market Cap vs. Volume')
    plt.xlabel('Market Cap')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)


    # end code


    st.header("Before we get into eahc part we will first visualize the data :")
    st.subheader("First lets see the scatter plot of all the coins :")
    st.pyplot(plt)

    st.header("Now we shall get into each part :")
    part_choise = st.selectbox("Which part shoild we go for ?", ["Part 1", "Part 2", "Part 3"])

    if part_choise == "Part 1":

            st.header("In this part we will use the KMeans algorithm with 5 clusters :")
            
            # code :

            features = data[['market_cap','volume']]
            model_kmeans = KMeans(n_clusters=5, random_state=42)
            features['cluster'] = model_kmeans.fit_predict(features)
            cluster_centroids = model_kmeans.cluster_centers_


            cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
            colorw=['lime','goldenrod','hotpink','greenyellow','cyan']
            plt.figure(figsize=(17,12))
            for cluster_id, color in cluster_colors.items():
                cluster_data = features[features['cluster'] == cluster_id]
                plt.scatter(cluster_data['market_cap'], cluster_data['volume'], label=f'Cluster {cluster_id+1}', c=color,s=75,alpha=0.5)
                plt.scatter(cluster_centroids[cluster_id, 0], cluster_centroids[cluster_id, 1], marker='*',s=120, c=colorw[cluster_id], label=f'Centroid {cluster_id+1}={cluster_centroids[cluster_id]}',alpha=1)

            plt.title('Coin Market Cap vs. Volume with K-Means Clustering')
            plt.xlabel('Market Cap')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True)


            # end code

            st.subheader("First we can see the centeroid for each cluster")
            cluster_names = [f"Cluster {i + 1}" for i in range(len(cluster_centroids))]
            df = pd.DataFrame(cluster_centroids, columns=["Centroid 1", "Centroid 2"], index=cluster_names)
            st.table(df)

            st.subheader("Now to visualize the cluster's with scatter plot :")
            st.pyplot(plt)

    if part_choise == "Part 2":

            st.subheader("In this part we are going to be model the Kmean with the k parameter ranging from 1 to 10 :")

            # code :

            # Define cluster colors

            inertias = []
            sil_score=[]

            ######################################################################################3

            features = data[['market_cap','volume']]
            # perform k-means clustering
            model_kmeans = KMeans(n_clusters=1, random_state=42)
            features['cluster'] = model_kmeans.fit_predict(features)
            #the cluster ids will be 0, 1, 2, 3, 4 for 5 clusters

            cluster_centroids = model_kmeans.cluster_centers_
            st.write('the cluster centroids will be in \n', cluster_centroids)
            cluster_colors = {0: 'red'}
            colorw=['lime']
            # Create a scatter plot with cluster centroids
            plt.figure(figsize=(17,12))
            for cluster_id, color in cluster_colors.items():
                cluster_data = features[features['cluster'] == cluster_id]
                fit_k=model_kmeans.fit(features)
                inertias.append(fit_k.inertia_)
                plt.scatter(cluster_data['market_cap'], cluster_data['volume'], label=f'Cluster {cluster_id+1}', c=color,s=75,alpha=0.5)
                plt.scatter(cluster_centroids[cluster_id, 0], cluster_centroids[cluster_id, 1], marker='*',s=120, c=colorw[cluster_id], label=f'Centroid {cluster_id+1}={cluster_centroids[cluster_id]}',alpha=1)

            # Customize the plot
            plt.title('Coin Market Cap vs. Volume with K-Means Clustering')
            plt.xlabel('Market Cap')
            plt.ylabel('Volume')
            plt.legend()

            # Show the plot
            plt.grid(True)
            st.pyplot(plt)

            cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange', 5:'yellow', 6:'cyan', 7:'black', 8:'brown', 9:'gray'}
            colorw=['k','y','r','cyan','m','hotpink','sienna','tan','b','g']

            for K in range(1,10):
                # perform k-means clustering
                features = data[['market_cap','volume']]
                model_kmeans = KMeans(n_clusters=K+1, random_state=42)
                features['cluster'] = model_kmeans.fit_predict(features)
                #the cluster ids will be 0, 1, 2, 3, ... , K for K clusters

                cluster_centroids = model_kmeans.cluster_centers_
                st.write(f'the cluster centroids  for k = {K+1} will be in \n', cluster_centroids)
                fit_k=model_kmeans.fit(features)
                inertias.append(fit_k.inertia_)
                sil_score.append(silhouette_score(features,fit_k.labels_,metric="euclidean",random_state=200))
                st.write("Silhouette score for k(clusters) = "+str(K+1)+" is "+str(silhouette_score(features,fit_k.labels_,metric="euclidean",random_state=200)))



                # Create a scatter plot with cluster centroids
                plt.figure(figsize=(17, 12))
                for cluster_id, color in cluster_colors.items():
                    if cluster_id > K:
                        break
                    cluster_data = features[features['cluster'] == cluster_id]
                    plt.scatter(cluster_data['market_cap'], cluster_data['volume'], label=f'Cluster {cluster_id+1}', c=color,alpha=0.65,s=75)
                    plt.scatter(cluster_centroids[cluster_id, 0], cluster_centroids[cluster_id, 1], marker='*', c=colorw[cluster_id], s=200, label=f'Centroid {cluster_id+1}={cluster_centroids[cluster_id]}')

                # Customize the plot
                plt.title('Coin Market Cap vs. Volume with K-Means Clustering')
                plt.xlabel('Market Cap')
                plt.ylabel('Volume')
                plt.legend()

                # Show the plot
                plt.grid(True)
                st.pyplot(plt)
            plt.figure(figsize=(17, 12))
            plt.plot(range(1,11),inertias,'bx-')
            plt.xlabel('value of k')
            plt.ylabel('inertia')
            plt.title('elbow method using inertia')
            st.subheader("Now the Elbow method using Inertia")
            st.pyplot(plt)
            sil_centers = pd.DataFrame({'Clusters' : range(2,11), 'Sil Score' : sil_score})
            st.write(sil_centers)
            sns.lineplot(x = 'Clusters', y = 'Sil Score', data = sil_centers, marker="+")
            st.subheader("Sil Score of Cluster's :")
            st.image("sil.png", caption="Sil Score of Clusters", use_column_width=True)         

            # end code

    if part_choise == "Part 3":
        st.header("In this part we are going to use DBScan to make model with 5 cluster's that has meaningfull info (based on 2 feature's (them being market cap and volume))")
        st.subheader("We are going to be experimenting on the parameters (number of neighbers and alpha) : ")
        
        st.header("First lets see the number of neighbers = 10")

        st.subheader("Some general information :")


        # code 
               
        n_neighbor=10
        neighbor,neighbor_index=NearestNeighbors(n_neighbors=n_neighbor).fit(data[['market_cap','volume']]).kneighbors()
        n=neighbor[:,n_neighbor-1]
        st.write('len:',len(n))
        st.write('max',max(n))
        st.write('min',min(n))
        sort_neighbor=np.sort(n)
        plt.plot(sort_neighbor)
        k_l=KneeLocator(np.arange(len(n)),sort_neighbor,curve='convex')
        st.write('knee',k_l.knee,'\nelbow',k_l.elbow)
        

        # end code


        st.subheader("The sorted neighbor plot :")
        st.pyplot(plt)
        st.subheader("The Knee point plot :")
        st.image("Knee.png", caption = "Knee point plot", use_column_width=True)

        # code :

        dbscan=DBSCAN(eps=16921911111.611433,min_samples=n_neighbor)
        dbscan.fit(data[['market_cap','volume']])
        plt.figure(figsize=(12, 6))
        plt.scatter(data['market_cap'], data['volume'], c=dbscan.labels_)
        plt.grid(True)
        plt.xlabel('MarketCap')
        plt.ylabel('Volume')
        plt.title('DBSCAN Clustering')


        np.unique(dbscan.labels_,return_counts=True)

        noise=list(dbscan.labels_).count(-1)
        noise = noise*100/len(dbscan.labels_)

        # end code 

        st.subheader("Market cap vs Volume scatter plot :")
        st.pyplot(plt)

        st.subheader("And the noise would be :")
        st.write(f"Noise: {noise}", key="number_box", format="0")


        st.header("Now to see the number of neighbers = 4")

        st.subheader("Some general information :")


        # code 
               
        n_neighbor=4
        neighbor,neighbor_index=NearestNeighbors(n_neighbors=n_neighbor).fit(data[['market_cap','volume']]).kneighbors()
        n=neighbor[:,n_neighbor-1]
        st.write('len:',len(n))
        st.write('max',max(n))
        st.write('min',min(n))
        sort_neighbor=np.sort(n)
        plt.plot(sort_neighbor)
        k_l=KneeLocator(np.arange(len(n)),sort_neighbor,curve='convex')
        st.write('knee',k_l.knee,'\nelbow',k_l.elbow)
        

        # end code


        st.subheader("The sorted neighbor plot :")
        st.pyplot(plt)
        time.sleep(5)
        
        st.subheader("The Knee point plot :")
        st.image(r"Knee_4.png", caption = "Knee3 point plot", use_column_width=True)
        
        #st.markdown('<img src="Kneee.png">', unsafe_allow_html=True)

        # code :

        from sklearn.cluster import DBSCAN
        dbscan=DBSCAN(eps=14100000000,min_samples=n_neighbor)#eps=0.2081727,min_samples=8
        dbscan.fit(data[['market_cap','volume']])
        time.sleep(2)
        plt.figure(figsize=(12, 6))
        plt.scatter(data['market_cap'], data['volume'], c=dbscan.labels_)
        plt.grid(True)
        plt.xlabel('MarketCap')
        plt.ylabel('Volume')
        plt.title('DBSCAN Clustering')


        np.unique(dbscan.labels_,return_counts=True)

        noise=list(dbscan.labels_).count(-1)
        noise*100/len(dbscan.labels_)

        # end code 

        st.subheader("Market cap vs Volume scatter plot :")
        st.pyplot(plt)

        st.subheader("And the noise would be :")
        st.write(f"Noise: {noise}", key="number_box", format="0")

        st.header("For this part we are going to alter the data to reduce the noise (and we are going to use number of neighbers = 10)")

        st.subheader("Some general information :")


        # code 
               
        G=data.loc[data['volume']<1*1e11]
        n_neighbor=10
        neighbor,neighbor_index=NearestNeighbors(n_neighbors=n_neighbor).fit(G[['market_cap','volume']]).kneighbors()
        n=neighbor[:,n_neighbor-1]
        st.write('len:',len(n))
        st.write('max',max(n))
        st.write('min',min(n))
        sort_neighbor=np.sort(n)
        plt.plot(sort_neighbor)
        k_l=KneeLocator(np.arange(len(n)),sort_neighbor,curve='convex')
        st.write('knee',k_l.knee,'\nelbow',k_l.elbow)
        

        # end code


        st.subheader("The sorted neighbor plot :")
        st.pyplot(plt)

        #st.subheader("The Knee point plot :")
        st.image(r"Knee_last.png", caption = "Knee2 point plot", use_column_width=True)

        # code :

        dbscan=DBSCAN(eps=16899999999,min_samples=n_neighbor)#eps=0.2081727,min_samples=8
        dbscan.fit(G[['market_cap','volume']])
        plt.figure(figsize=(12, 6))
        plt.scatter(G['market_cap'], G['volume'], c=dbscan.labels_)
        plt.grid(True)
        plt.xlabel('MarketCap')
        plt.ylabel('Volume')
        plt.title('DBSCAN Clustering')


        np.unique(dbscan.labels_,return_counts=True)

        noise=list(dbscan.labels_).count(-1)
        noise = noise*100/len(dbscan.labels_)

        # end code 

        st.subheader("Market cap vs Volume scatter plot :")
        st.pyplot(plt)

        st.subheader("And the noise would be :")
        st.write(f"Noise: {noise}", key="number_box", format="0")



if report_choice == "Report 2":


    st.header("Report 2")
    data = pd.read_excel(r'coins_data.xlsx')

    st.subheader("Which part are we going for ?")
    part_choise = st.selectbox("", ["Part 1", "Part 2", "Part 3", "Part 4"])



    if part_choise == "Part 1":
        st.header("In this part we use hierarchical clustering with 2 cluster's and 2 features.")
        st.write("First we should take a look at the Dendogram we drew for this part :")


        # Code :
    
        coin_data = pd.read_excel(r'coins_data.xlsx')
        features = coin_data[['MarketCap', 'Volume']]
        coin_names = coin_data['Symbol']

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        from scipy.cluster.hierarchy import dendrogram, linkage
        from plotly.offline import iplot
        import plotly.figure_factory as ff


        Z = sch.linkage(features, method='ward')
        plt.figure(figsize=(10, 6))
        dendrogram = sch.dendrogram(Z, labels=coin_names.tolist(), orientation='top')

        plt.title('Dendrogram')
        plt.xlabel('Cryptocurrencies')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)

        # end code

        st.pyplot(plt)


        st.header("First a general explanation on the Dendogram:")
        st.write("""Ok so the dendogram graph is shown above, but what does it mean ?
        the dendogram is generaly going tp show use the data as hierarchical clusters, meaning at the start each observation is a point in the x_axis 
        and as we come up in the dendogram, these data points join togheter to form the clusters at their respective levels.
        in this project we are given the number of clusters, but for completion sake we are going to try to figure out the optimal amount of clusters.
        the manner in which we choose the number of clusters in a dendogram is rather instinctive, but there is a way to start quesing.
        we have to draw a horizantal line through the plot, and however many colisions we have with the lines in the plot, are the number of clusters.
        in this case i think we are better off having 3 clusters. (after the second part we will discuss this fully)
        """)


        # code :

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)


        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")


        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""This cluster includes the cryptocurrencies "USDT" and "BTC." 
                    These Cryptocurrencies stand out as the cryptocurrencies with the highest market capitalization and trading volume among the 20 cryptocurrencies.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cluster 2 is the larger cluster, containing a wide range of cryptocurrencies, including "LEO," "UNI," "WBTC," "AVAX," "DAI," "SHIB," "LTC," "TRX," "DOT," "MATIC," "SOL," "DOGE," "ADA," "BUSD," "XRP," "USDC," "BNB," and "ETH."
                    These cryptocurrencies generally have lower market capitalization and trading volume compared to those in Cluster 0.""")
        
        st.subheader("Generally speaking")
        st.write("""The model has grouped cryptocurrencies based on their market cap and volume. 
                    As expected it has separated Bitcoin and USDT, which are notably different from the rest of the cryptocurrencies due to their much higher market capitalization and trading volume.
                    If we were to need more meaningfull clusters or different groupings, we could experiment with different clustering techniques, features, or adjust the number of clusters based on our specific goal.""")
        

    if part_choise == "Part 2":
        st.header("In this part we use hierarchical clustering with 2 cluster's but with 3 features.")
        st.write("First we should take a look at the Dendogram we drew for this part :")

        # Code :
    
        coin_data = pd.read_excel(r'coins_data.xlsx')
        coin_names = coin_data['Symbol']

        from sklearn.preprocessing import LabelEncoder
        from sklearn.cluster import AgglomerativeClustering

        coin_data = pd.get_dummies(coin_data, columns=['ProofType'], prefix='ProofType')

        features = coin_data[['MarketCap', 'Volume'] + [col for col in coin_data.columns if col.startswith('ProofType_')]]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)

        from scipy.cluster.hierarchy import dendrogram, linkage
        from plotly.offline import iplot
        import plotly.figure_factory as ff

        Z = sch.linkage(features, method='ward')
        plt.figure(figsize=(10, 6))
        dendrogram = sch.dendrogram(Z, labels=coin_names.tolist(), orientation='top')

        plt.title('Dendrogram')
        plt.xlabel('Cryptocurrencies')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)

        # end code

        st.pyplot(plt)

        # code :

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)


        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")



        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""Cryptocurrencies: ['WBTC', 'DAI', 'LTC', 'DOT', 'SOL', 'DOGE', 'BUSD', 'XRP', 'USDC', 'USDT', 'BTC']

Interpretation: This cluster seems to consist of cryptocurrencies with relatively lower market capitalization and volume. 

Most of these cryptocurrencies are not among the top 10 by market cap. 

They include stablecoins like 'USDT' and 'USDC,' which are known for their price stability and are often used as trading pairs.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cryptocurrencies: ['LEO', 'UNI', 'AVAX', 'SHIB', 'TRX', 'MATIC', 'ADA', 'BNB', 'ETH']

Interpretation: Cluster 2 appears to contain cryptocurrencies with higher market capitalization and trading volume. 

These are some of the top cryptocurrencies in terms of market cap and are associated with well-known platforms like Ethereum ('ETH'), Cardano ('ADA'), and Binance Coin ('BNB').""")

        st.header("So what conclusions can we draw from these results ?")
        st.write("""Keeping the number of clusters constant at two, including the "ProofType" feature has resulted in more refined clusters. 

Separating cryptocurrencies into two clusters based on their market capitalization, volume, and ProofType. 

In the first part, the separation is primarily between a stablecoin ('USDT') and Bitcoin ('BTC').
                 
In summary, the second part provides a more detailed and clear view of the cryptocurrency market, considering 'ProofType' features in clustering. 

It allows for a better understanding of the various cryptocurrencies and their groupings based on these features. 

The first part is simpler and divides cryptocurrencies into two broad categories, primarily based on 'USDT' and 'BTC.' 

The choice of which approach to use depends on the specific analysis and insights we aim to derive from the data.""")




    if part_choise == "Part 3":
        st.header("In this part we use hierarchical clustering with 3 features but with 3 clusters as well (experimenting on the number of clusters)")
        st.write("First we should take a look at the Dendogram we drew for this part :")

        # Code :
    
        coin_data = pd.read_excel(r'coins_data.xlsx')
        coin_names = coin_data['Symbol']

        coin_data = pd.get_dummies(coin_data, columns=['ProofType'], prefix='ProofType')

        features = coin_data[['MarketCap', 'Volume'] + [col for col in coin_data.columns if col.startswith('ProofType_')]]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        n_clusters = 3
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)


        from scipy.cluster.hierarchy import dendrogram, linkage
        from plotly.offline import iplot
        import plotly.figure_factory as ff

        Z = sch.linkage(features, method='ward')
        plt.figure(figsize=(10, 6))
        dendrogram = sch.dendrogram(Z, labels=coin_names.tolist(), orientation='top')

        plt.title('Dendrogram')
        plt.xlabel('Cryptocurrencies')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)

        # end code

        st.pyplot(plt)

        # code :

        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]
        cluster_2 = coin_data[coin_data['Cluster'] == 2]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()
        cluster_2_names = cluster_2['Symbol'].tolist()


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the third one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")



        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""Cryptocurrencies: ['WBTC', 'LTC', 'DOT', 'SOL', 'DOGE', 'XRP', 'BTC']

Interpretation: Cluster 1 consists of cryptocurrencies that have relatively high market capitalization and volume. 

These cryptocurrencies are significant players in the market and include well-known names like Bitcoin ('BTC') and Ripple ('XRP').""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cryptocurrencies: ['LEO', 'UNI', 'AVAX', 'SHIB', 'TRX', 'MATIC', 'ADA', 'BNB', 'ETH']

Interpretation: Cluster 2 includes cryptocurrencies with high market capitalization and trading volume. 

These cryptocurrencies are among the top performers in the market, and they represent a wide range of use cases, including decentralized finance (DeFi), smart contract platforms, and meme coins.""")
        
        st.subheader("What about cluster 3 ?")
        st.write("""Cryptocurrencies: ['DAI', 'BUSD', 'USDC', 'USDT']

Interpretation: Cluster 3 primarily consists of stablecoins. Stablecoins like 'USDT,' 'USDC,' 'DAI,' and 'BUSD' are designed to maintain a stable value and are widely used for trading and as a store of value in the cryptocurrency market.

These cryptocurrencies are associated with the 'stablecoin' Proof Type.""")



        st.subheader("So ?")
        st.write("""This three-cluster approach provides more granularity in categorizing cryptocurrencies. 

It further separates the stablecoins into their own cluster (Cluster 3), highlighting their distinct characteristics compared to other cryptocurrencies.

Clusters 1 and 2 still represent cryptocurrencies with high market capitalization and trading volume, but the division between them is more pronounced, possibly reflecting differences in the market dynamics or use cases among these cryptocurrencies.

The choice of three clusters may be useful when we want to distinguish between stablecoins and other types of cryptocurrencies, providing a more detailed analysis of the market. 

However, the specific interpretation of the clusters may vary based on the context and the goals of our analysis.""")




    if part_choise == "Part 4":
        st.header("In this part we use hierarchical clustering with 2 clusters but with more features (we want to see the effect of the number of features here)")
        st.write("First we should take a look at the Dendogram we drew for this part as well :")

        # Code :
    
        coin_data = pd.read_excel(r'coins_data.xlsx')
        coin_names = coin_data['Symbol']

        coin_data = pd.get_dummies(coin_data, columns=['ProofType'], prefix='ProofType')

        features = coin_data[['MarketCap', 'Volume'] + [col for col in coin_data.columns if col.startswith('ProofType_')]]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        coin_data = pd.get_dummies(coin_data, columns=['Network'], prefix='Network')

        features = coin_data[['MarketCap', 'Volume', 'ProofType_PoH','ProofType_PoS','ProofType_PoW','ProofType_RPCA','ProofType_stablecoin'] + [col for col in coin_data.columns if col.startswith('Network_')]]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        from scipy.cluster.hierarchy import dendrogram, linkage
        from plotly.offline import iplot
        import plotly.figure_factory as ff

        Z = sch.linkage(features, method='ward')
        plt.figure(figsize=(10, 6))
        dendrogram = sch.dendrogram(Z, labels=coin_names.tolist(), orientation='top')

        plt.title('Dendrogram')
        plt.xlabel('Cryptocurrencies')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)

        # end code

        st.pyplot(plt)

        # code :

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)

        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()

        print("Cryptocurrencies in Cluster 0:")
        print(cluster_0_names)

        print("\nCryptocurrencies in Cluster 1:")
        print(cluster_1_names)


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")


        st.header("Conclusion :")
        st.subheader("Cluster 1 :")
        st.write("""Cryptocurrencies: ['LEO', 'UNI', 'WBTC', 'AVAX', 'DAI', 'SHIB', 'LTC', 'TRX', 'DOT', 'MATIC', 'SOL', 'DOGE', 'ADA', 'BUSD', 'XRP', 'USDC', 'BNB', 'USDT', 'ETH']

Interpretation: Cluster 1 appears to include a wide range of cryptocurrencies with various use cases, market capitalization, and trading volume. 

This cluster is more diverse and contains cryptocurrencies representing decentralized finance (DeFi), meme coins, stablecoins, and major platforms such as Ethereum ('ETH') and Binance Coin ('BNB'). 

It seems to encompass a broad spectrum of the cryptocurrency market.""")

        st.subheader("Cluster 2 :")
        st.write("""Cryptocurrencies: ['BTC']

Interpretation: Cluster 2 includes only one cryptocurrency, which is Bitcoin ('BTC'). 

Bitcoin is unique in this clustering, being separate from other cryptocurrencies. 

This might indicate that Bitcoin stands out from the rest in terms of market capitalization, trading volume, and network.""")
        

        st.subheader("So how did the additional feature do ?")
        st.write("""The addition of the 'Network' feature further refines the clustering by taking into account the underlying blockchain networks of cryptocurrencies. 

In this particular case, Cluster 1 contains a diverse mix of cryptocurrencies, while Cluster 2 highlights the exceptional position of Bitcoin. 

This clustering provides insights into how Bitcoin differs from the rest of the cryptocurrency market due to its historical significance and distinct characteristics.

But it would be a more refined analysis if the number of clusters could be more.

As we can see from the dendogram, the model could benefit from additional clusters.""")





if report_choice == "Report 3":

    # code :

    train = pd.read_csv(r'train.csv')
    teste = pd.read_csv(r'teste.csv')
    testf = pd.read_csv(r'testf.csv')



    # end code


    st.header("Report 3")

    st.subheader("Choose a Dataset to Display:")
    dataset_choice = st.selectbox("Select a Dataset", ["Train", "Teste", "Testf", "New df"])

    if dataset_choice == "Train":
        data = pd.read_csv(r'train.csv')
    elif dataset_choice == "Teste":
        data = pd.read_csv(r'teste.csv')
    elif dataset_choice == "Testf":
        data = pd.read_csv(r'testf.csv')
    elif dataset_choice == "New df":
        data = pd.read_csv(r"New df.csv")

    st.write(f"Displaying {dataset_choice} Dataset:")
    st.dataframe(data)

    st.write("First we prep the data (check for imbalancement and decomposite the data and reduce dimensionallity and standardizing it)")


    st.header("We created 3 models for this part compared them and chose the best one. (and a little problem at the end)")

    model_choice = st.selectbox("Which model would you like to see ?", ["Model 1", "Model 2", "Model 3", "Big Boss"])


    if model_choice == "Model 1":

        st.header("For the first model we used the KNN method :")
        st.subheader("Here we have 3 stat's (Scatter Plot, Heat Map, Score) :")

        st.write("Scatter Plot :")
        st.image(r"SP_KNN_EV.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_KNN_EV.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.5769230769230769
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.5178571428571428
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.52
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.5178571428571428
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.5555555555555555
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.write("The F1 score for this model is about 0.55, and so this method is not so appealing.")


    elif model_choice == "Model 2":

        
        st.header("For the Second model we used the Random Forest method :")
        st.subheader("Lets see out stats :")

        st.write("Scatter Plot :")
        st.image(r"SP_RF_EV.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_RF_EV.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.5897435897435898
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.8214285714285714
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.58
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.547077922077922
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.6865671641791046
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.write("The F1 score for this method is about 0.68, and compared to the previous one and in general it is accaptable for now.")


        st.header("So far we have used the eval data to find the F_1 score and to compare, but now that we have chosen the method we are gonna use, we might aswell use the actuall test data and see the F1 score for it.")
        st.subheader("Lets see out stats :")


        st.write("Scatter Plot :")
        st.image(r"SP_RF_TE.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_RF_TE.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.6071428571428571
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.9444444444444444
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.6
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.513888888888889
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.7391304347826088
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.write("We can see that the F1 score has gone up compered to before, this could be because of the fact that our two sets are located in different trend intervals and therefore have different distribution of 0s and 1s.")

        st.header("For ADHD sake, we are going to implement a backtracking system :")

        st.write("Scatter Plot :")
        st.image(r"SP_RF_BT.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_RF_BT.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.5555555555555556
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.8333333333333334
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.5
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.4166666666666667
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.6666666666666667
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.write("""So, if we use a 30 day interval as trend, we can find the next trend with about 74% accuracy on both 1s and 0s, but if we try to do this mid-trends and one by one like our backtesting model; we will see a little bit of setback because the model is not capabale of finding itself's location in the respective trend.

Therefore our backtesting system's accuracy on both 1s and 0s would be about 67%.""")


    elif model_choice == "Model 3":
        
        st.header("As we saw, the f1 score for the randoom forest is acceptable but we can use more complex models such as AdaBoostClassifier :")

        st.subheader("Lets see out stats :")

        st.write("Scatter Plot :")
        st.image(r"SP_ADA_EV.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_ADA_EV.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.625
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.7142857142857143
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.6
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.5844155844155845
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.6666666666666666
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")



        st.write("Not very apealing, but we might as well use out test data aswell.")

        st.subheader("Lets see out stats :")

        st.write("Scatter Plot :")
        st.image(r"SP_RF_TE.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_RF_TE.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.5925925925925926
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.8888888888888888
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.5666666666666667
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.4861111111111111
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.711111111111111
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.write("Let's see how the back testing works :")
    
        st.subheader("Lets see out stats :")

        st.write("Scatter Plot :")
        st.image(r"SP_RF_BT.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"HM_RF_BT.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.5769230769230769
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.8333333333333334
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.5333333333333333
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.45833333333333337
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.6818181818181818
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")

 
        st.write("""As you can see, AdaBoost is more consistent with Backtesting which is a better measure for our F1 Score because of a better distribution of 1s and 0s.
                    Finally we can say our RandomForest is better in a full trend finding way with a F1-score of 74% and has a pretty good backtesting F1-score of 67% approximately.
                    Also our AdaBoost model is slightly better in backtesting with more than 68% in F1-score.""")


    if model_choice == "Big Boss":
        st.header("The final question :")
        st.subheader("Here (as you can quess by the title) we have created a problem to be solved :")
        st.subheader("""The new question is 'Is Bitcoin sensitive to worldwide news and people opinions?'.

Now we will try to use sentiment analysis on Bitcoin's wikipedia page to answer that question:""")

        st.write("Here we are going to use the New df data fram (you can view this data set at the top)")

        st.subheader("Now we will use the Adaboost and the respective backtesting system that we have used before :")

        st.subheader("Lets see out stats (for the eval data) :")

        st.write("Scatter Plot :")
        st.image(r"TW_ADA_SP_1.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"TW_ADA_HM_1.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.4117647058823529
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.6666666666666666
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.46
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.48850574712643674
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.509090909090909
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")



        st.subheader("Lets see out stats (for the test data this time) :")

        st.write("Scatter Plot :")
        st.image(r"TW_ADA_TE_SP.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"TW_ADA_TE_HM.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.42105263157894735
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.5333333333333333
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.4
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.4
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.47058823529411764
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.subheader("Lets see out stats (back testing (ADHD)) :")

        st.write("Scatter Plot :")
        st.image(r"TW_ADA_BT_SP.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"TW_ADA_BT_HM.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.42105263157894735
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.5333333333333333
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.4
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.4
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.47058823529411764
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.write("As you can see, bitcoin and the news source that we have selected has not much in common and therefore the result is weaker than even randomly distributing 1s and 0s!")

        st.header("Let's see if using other complex models will help us reaching better results:")
        st.subheader("Lets see out stats (for the ecal first) :")

        st.write("Scatter Plot :")
        st.image(r"TW_CLA_SP_1.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"TW_CLA_HM_1.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.4838709677419355
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.7142857142857143
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.56
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.58128078817734
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.5769230769230769
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")


        st.subheader("Lets see out stats (for the test next) :")

        st.write("Scatter Plot :")
        st.image(r"TW_CLA_SP_TE.png", caption = "Scatter Plot", use_column_width=True)

        st.write("Heat Map :")
        st.image(r"TW_CLA_HM_TE.png", caption = "Heat Map", use_column_width=True)

        precision_score = 0.47058823529411764
        st.write(f"Precision Score : {precision_score}", key="number_box", format="0")

        recall_score = 0.5333333333333333
        st.write(f"Recall Score : {recall_score}", key="number_box", format="0")

        accuracy_score = 0.4666666666666667
        st.write(f"Accuracy Score : {accuracy_score}", key="number_box", format="0")

        roc_auc_score = 0.4666666666666667
        st.write(f"Roc Auc Score : {roc_auc_score}", key="number_box", format="0")

        f1_score = 0.5
        st.write(f"F1 Score: {f1_score}", key="number_box", format="0")



        st.subheader("Even trying NN didn't help us much, therefore we come to the result that our source (wikioedia revesions) doesen't have the proper correlation with bitcoin prices.")
