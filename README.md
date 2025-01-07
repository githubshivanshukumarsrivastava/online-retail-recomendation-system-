# online-retail-recomendation-system-


#STEP 1 => IMPORT LIBRARIES

import pandas as pd
import numpy as numpy
import matplotlib. pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#import numpy as numpy
import os
from scipy.sparse import coo_matrix
import datetime as dt
from sklearn.cluster import KMeans
import random

#step 2=load dataset and read it 

train_data=pd.read_excel("C:\\Users\\Computer Care\\Downloads\\OnlineRetail(1).xlsx","OnlineRetail")
train_data.columns
train_data.head()
train_data.tail()
train_data.info()
train_data = train_data[train_data['CustomerID'].notnull()]
train_data.shape
train_data.isnull().sum()
train_data.info()


#step 3=optimizing data for enhanced clustering 
 We did this so we would have the date of each invoice distinctly on daily basis
in our data. This is important because as the data is going to be stored and
analyzed based on different days and often the data is grouped according to
different days. Thus, the presence of the (`InvoiceDay`) column enables time-
based analyses that are more diverse than the rates specified by the cardinality
of a month.


train_data['InvoiceDay'] = train_data['InvoiceDate'].apply(lambda x:dt.datetime(x.year, x.month, x.day))
train_data.head()



"Finding Last Purchase Date for Customer Management":
Now, we want to find the last purchase date of customers to perform proper
clustering. This is because the date of the customers' last purchase can be a
crucial point for applications or marketing strategies. Therefore, these codes
are used for better customer management and planning, allowing us to cluster
customers effectively based on their last purchase date.


dt.timedelta(1)
pin_date = max(train_data['InvoiceDay']) + dt.timedelta(1)
pin_date


"creating 'totalsum'variable for financial analysis ":

train_data['TotalSum'] = train_data['Quantity'] * train_data['UnitPrice']
train_data.head()

creating RFM variable for customr analysis and marketing stratigies:

rfm = train_data.groupby('CustomerID').agg({
'InvoiceDay': lambda x: (pin_date - x.max()).days,
'InvoiceNo': 'count',
'TotalSum': 'sum'
})
rfm


# To sum up,
this code was produced with intention of creating for each customer Recency,
Frequency, Monetary variables known as RFM. These three variables hold
significant importance in customer analysis and marketing strategies:
I. Recency (R): This variable informs us on the days that have elapsed since the
last purchase that was made by the customer. Such means, we can calculate the
number of days since the last purchase and define customers’ activity level and
additional influencing them if needed.
2. Frequency (F): This variable holds information about the total number of
purchases that has been done by the customer. Such information assists us in
ascertaining individuals who frequently patronize our products and services;
therefore, we offer them some promos and coupons.
3. Monetary (M): This variable is used to indicate the total expenditure of the
customer to be able to make any purchases. And it helps to identify customers who
spend more and assign them to the valuable and prospective customers group. With
the help of this code, it is easy to group the customers and classify them such
as according to the recent activity of the customer, how frequently the customer
has made the purchases and how much the customer spends. It also helps classify
our decision making in the areas of marketing and advertising so as to make
proper customer centric strategies.#


rfm.rename(columns= {
'InvoiceDay': 'Recency',
'InvoiceNo': 'Frequency',
'TotalSum': 'Monetary'
}, inplace=True)
rfm


step 4:data preprocessing 
r_labels = range(4, 0, -1) #[4, 3, 2, 1]
r_groups = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
f_labels = range(1, 5) # [1, 2, 3, 4]
f_groups = pd.qcut(rfm['Frequency'], q=4, labels=f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
rfm['R'] = r_groups.values
rfm['F'] = f_groups.values
rfm['M'] = m_groups.values
rfm


step 5= customer clustering for target market 


X = rfm[['R', 'F', 'M']]
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300)
kmeans.fit(X)
kmeans.labels_
rfm['kmeans_cluster'] = kmeans.labels_rfm


step 6=customer clustering viualization 


# Number of clusters
num_clusters = 10


# Create subplots with two clusters in each row
fig, axes = plt.subplots(num_clusters // 2, 2, figsize=(12, 20))


# Flatten the axes array to iterate through subplots
axes = axes.ravel()


# Loop through each cluster and plot it
for cluster_id in range(num_clusters):


# Filter data for the current cluster
    cluster_data = rfm[rfm['kmeans_cluster'] == cluster_id]


# Plot the data with a distinct color
    sns.scatterplot(data=cluster_data, x='Recency', y='Frequency',
hue='Monetary', palette='viridis', ax=axes[cluster_id])


# Set the title for the subplot
    axes[cluster_id].set_title(f'Cluster {cluster_id}')


# Customize axes labels, if needed
# axes[cluster_id].set_xlabel('X-axis Label')
# axes[cluster_id].set_ylabel('Y-axis Label')

# Add a common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()
create histogram
plt.figure(figsize=(12, 6))
for cluster_id in range(num_clusters):
    plt.subplot(2, 5, cluster_id + 1)
    sns.histplot(rfm[rfm['kmeans_cluster'] == cluster_id]['Recency'], bins=20, kde=True)
    plt.title(f'Cluster {cluster_id}')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show()
step 7:create recomendation system 


# Number of clusters (groups)
num_clusters = 10

# Create an empty dictionary to store recommendations for each cluster
cluster_recommendations = {}

# Loop through each cluster
for cluster_id in range(num_clusters):
    # Find customers in the current cluster
    customers_in_cluster = rfm[rfm['kmeans_cluster'] == cluster_id].index
    
    # Find top products for customers in the current cluster
    top_products_for_cluster = train_data[train_data['CustomerID'].isin(customers_in_cluster)].groupby(['StockCode'])['InvoiceNo'].count().sort_values(ascending=False).head(10)
    
    # Store the top products for the current cluster in the dictionary
    cluster_recommendations[f'Cluster {cluster_id}'] = top_products_for_cluster.index.tolist()

# Display the recommendations for each cluster
for cluster, recommended_products in cluster_recommendations.items():
    print(f"{cluster} -> Recommended Products: {recommended_products}")
cluster recomendation system
def generate_cluster_recommendations(num_clusters, num_customers_to_display, rfm, train_data):
    # Create an empty dictionary to store recommendations for each cluster
    cluster_recommendations = {}

    # Loop through each cluster
    for cluster_id in range(num_clusters):
        # Find customers in the current cluster
        customers_in_cluster = rfm[rfm['kmeans_cluster'] == cluster_id].index

        # Find top products for customers in the current cluster
        top_products_for_cluster = train_data[train_data['CustomerID'].isin(customers_in_cluster)].groupby(['StockCode'])['InvoiceNo'].count().sort_values(ascending=False).head(10)

        # Find customers who haven't purchased any of the top products in the current cluster
        non_buyers = [customer for customer in customers_in_cluster if not (train_data[(train_data['CustomerID'] == customer) & (train_data['StockCode'].isin(top_products_for_cluster.index.tolist()))]).empty]


        # Limit the number of non-buyers to the specified number
        num_customers_to_display = min(num_customers_to_display, len(non_buyers))

        # Select non-buyer customers for the current cluster
        selected_customers = non_buyers[:num_customers_to_display]

        # Store the top products and selected non-buyer customers for the current cluster in the dictionary
        cluster_recommendations[f'Cluster {cluster_id}'] = {
            'Recommended Products': top_products_for_cluster.index.tolist(),
            'Selected Non-Buyer Customers': selected_customers
        }

    return cluster_recommendations

# Example usage:
num_clusters = 10
num_customers_to_display = 5

# Assuming you already have 'rfm' and 'df' dataframes
cluster_recommendations = generate_cluster_recommendations(num_clusters, num_customers_to_display, rfm, train_data)

# Display the recommendations and selected non-buyer customers for each cluster
for cluster, recommendations_and_customers in cluster_recommendations.items():
    print(f"{cluster} ->")
    print("Recommended Products:")
    for customer_id in recommendations_and_customers['Selected Non-Buyer Customers']:
        print(f"Customer: {customer_id} =====>>>> Recommended Products: {recommendations_and_customers['Recommended Products']}")
    print()
