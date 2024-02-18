from django.shortcuts import render
from plotly.offline import plot
import plotly.express as px
import io
import base64
from PIL import Image


# Organize imports alphabetically
from datetime import datetime
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('TkAgg')  # Use Agg backend with Tkinter for rendering
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.http import HttpResponse
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, MeanShift, AffinityPropagation, OPTICS, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from plotnine import ggplot, aes, geom_line, geom_point, facet_wrap, scale_x_date, theme, scale_x_continuous, geom_col, coord_flip
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def import_dataset():
    data_type = {
        '_CustomerID': str
    }
    sales_data = pd.read_csv("C:/Users/Dell/Downloads/sales_data.csv", dtype=data_type, parse_dates=['OrderDate'])
    
    # Calculate revenue
    sales_data['Revenue'] = (sales_data['Unit Price'] - (sales_data['Unit Price'] * sales_data['Discount Applied']) -
                             sales_data['Unit Cost']) * sales_data['Order Quantity']
    
    # Calculate mean revenue for each customer
    mean_revenue = sales_data.groupby('_CustomerID')['Revenue'].transform('mean')

     # Create a new column 'MeanRevenue' in cutoff_in DataFrame
    sales_data['MeanRevenue'] = mean_revenue
    
    return sales_data

def plot_sales_data(sales_data):
    sales_data1 = (
        sales_data
        .reset_index()
        .set_index('OrderDate')
        [['Revenue']]
        .resample('MS')
        .sum()
    )
    fig = sales_data1.plot(figsize=(12, 7)).get_figure()
    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()
    # Close the buffer
    buffer.close()
    # Return the base64 encoded image string
    return img_str

def home(request):
    context_dict = {
        'name': 'Abhinav'

    }
    return render(request, 'segmentation_app/segmentation2.html', context_dict)

def histogram_make(sales_data):
    # Create a new figure
    fig = plt.figure(figsize=(12, 7))

    # Plot the histogram
    plt.hist(sales_data['Order Quantity'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Order Quantity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Order Quantity')
    
    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()

    # Close the buffer
    buffer.close()

    # Return the base64 encoded image string
    return img_str

def pie_chart(sales_data):
    # Create a new figure

# Calculate total revenue for each sales channel
    revenue_by_channel = sales_data.groupby('Sales Channel')['Revenue'].sum()
    fig = plt.figure(figsize=(12, 7))

# Plot
    revenue_by_channel.plot(kind='pie', autopct='%1.1f%%', colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen'])
    plt.title('Revenue Distribution Across Sales Channels')
    plt.ylabel('')
    
    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()

    # Close the buffer
    buffer.close()

    # Return the base64 encoded image string
    return img_str

def revenue_trend(sales_data):

# Group sales data by OrderDate and sum the revenue
    revenue_over_time = sales_data.groupby('OrderDate')['Revenue'].sum()
    fig = plt.figure(figsize=(18, 10.5))


# Calculate total revenue for each sales channel
# Plot
    revenue_over_time.plot(color='blue')
    plt.title('Revenue Trend Over Time')
    plt.xlabel('Order Date')
    plt.ylabel('Revenue')
    plt.grid(True)
    
    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()

    # Close the buffer
    buffer.close()

    # Return the base64 encoded image string
    return img_str

def order_quantity_over_time(sales_data):
    # Group sales data by OrderDate and sum the order quantity
    order_quantity_over_time = sales_data.groupby('OrderDate')['Order Quantity'].sum()

# Group sales data by OrderDate and sum the revenue
    fig = plt.figure(figsize=(18, 10.5))
    order_quantity_over_time.plot(color='green')
    plt.title('Order Quantity Trend Over Time')
    plt.xlabel('Order Date')
    plt.ylabel('Order Quantity')
    plt.grid(True)
    
    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()

    # Close the buffer
    buffer.close()

    # Return the base64 encoded image string
    return img_str

def customer_segmentation(sales_data):
    df = sales_data.copy()
    df['OrderDate'].isnull().sum()

    df.dtypes

    columns = ['OrderNumber', '_CustomerID', 'OrderDate', 'Revenue', 'MeanRevenue']
    df_dataset = df[columns]

    df_dataset

    n_days = 180
    max_date = df_dataset['OrderDate'].max()
    cutoff = max_date - pd.to_timedelta(n_days, unit="d")
    cutoff1 = cutoff - pd.to_timedelta(n_days, unit="d")

    cutoff_in = df_dataset[df_dataset['OrderDate'] <= cutoff].copy()
    cutoff_out = df_dataset[df_dataset['OrderDate'] > cutoff]
    cutoff_in1 = df_dataset[(df_dataset['OrderDate'] <= cutoff) & (df_dataset['OrderDate'] > cutoff1)]

    # Display cutoff_in DataFrame
    cutoff_in1

    # Assuming targets_df is defined
    targets_df = cutoff_out.groupby('_CustomerID')['Revenue'].sum().reset_index()\
        .rename(columns={'Revenue': 'Spent_last_30'})\
        .assign(Spent_last_30_flag=1)

    # Set '_CustomerID' as the index
    targets_df.set_index('_CustomerID', inplace=True)

    # Display targets_df
    targets_df

    features_df = cutoff_in.groupby('_CustomerID').agg({
        'OrderDate': lambda v: (cutoff_in['OrderDate'].max() + pd.Timedelta(days=1) - v.max()).days,
        'OrderNumber': 'count',
        'Revenue': 'sum',
        'MeanRevenue': 'first'
    })
    features_df1 = cutoff_in1.groupby('_CustomerID').agg({
        'OrderDate': lambda v: (cutoff_in['OrderDate'].max() + pd.Timedelta(days=1) - v.max()).days,
        'OrderNumber': 'count',
        'Revenue': 'sum',
        'MeanRevenue': 'first'
    })
    features_df1

    features_df.rename(
        columns={
            'OrderDate': 'Recency_before_last_30_days',
            'OrderNumber': 'Frequency_before_last_30_days',
            'Revenue': 'Monetary_before_last_30_days',
            'MeanRevenue': 'Average_Money_before_last_30_days'
        },
        inplace=True
    )
    features_df1.rename(
        columns={
            'OrderDate': 'Recency_before_180_days_of_cutoff',
            'OrderNumber': 'Frequency_before_180_days_of_cutoff',
            'Revenue': 'Monetary_before_180_days_of_cutoff',
            'MeanRevenue': 'Average_Money_before_180_days_of_cutoff'
        },
        inplace=True
    )
    features_df1

    features_df = features_df.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)
    features_df1 = features_df1.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)

    features_df

    df_dataset['LengthOfUsage'] = (df_dataset.groupby('_CustomerID')['OrderDate'].transform('max') -
                                   df_dataset.groupby('_CustomerID')['OrderDate'].transform('min')).dt.days.copy()

    today_date = pd.to_datetime('2021-01-01')

    lrfm_dataset = df_dataset.groupby('_CustomerID').agg({
        'OrderDate': lambda v: (today_date - v.max()).days,
        'OrderNumber': 'count',
        'Revenue': 'sum',
        'LengthOfUsage': 'first'  # Length of Usage
    })
    lrfm_dataset

    lrfm_dataset.rename(
        columns={
            'OrderDate': 'Recency',
            'OrderNumber': 'Frequency',
            'Revenue': 'Monetary',
            'LengthOfUsage': 'Length'
        },
        inplace=True
    )
    lrfm_dataset

    lrfm_dataset['R'] = np.where(lrfm_dataset['Recency'] < 4.001, 1,
                                 np.where((4.001 <= lrfm_dataset['Recency']) & (lrfm_dataset['Recency'] <= 8.000), 2, 3))

    lrfm_dataset['F'] = np.where(lrfm_dataset['Frequency'] < 152.333, 1,
                                 np.where((152.333 <= lrfm_dataset['Frequency']) & (lrfm_dataset['Frequency'] <= 164.00),
                                          2, 3))

    lrfm_dataset['M'] = np.where(lrfm_dataset['Monetary'] < 402127.000, 1,
                                 np.where((402127.000 <= lrfm_dataset['Monetary']) & (lrfm_dataset['Monetary'] <= 443110),
                                          2, 3))

    lrfm_dataset['L'] = np.where(lrfm_dataset['Length'] < 200.00, 1,
                                 np.where((500.00 <= lrfm_dataset['Length']) & (lrfm_dataset['Length'] <= 900.00), 2,
                                          3))

    lrfm = lrfm_dataset
    lrfm

    lrfm['lrfm_group'] = lrfm[['L', 'R', 'F', 'M']].apply(lambda v: '-'.join(v.astype(str)), axis=1)

    lrfm[['L', 'R', 'F', 'M']] = lrfm[['L', 'R', 'F', 'M']]
    lrfm['lrfm_score_total'] = lrfm[['L', 'R', 'F', 'M']].sum(axis=1)

    LRFM = lrfm.copy()

    LRFM = LRFM.drop(columns=['lrfm_group'])

    LRFM['lrfm'] = LRFM['R'] * 1000 + LRFM['F'] * 100 + LRFM['M'] * 10 + LRFM['L']

    LRFM

    Z = LRFM[['L', 'R', 'F', 'M']]

    # Choose the number of clusters (you can tune this parameter)
    n_clusters = 30

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, min_samples=1)  # You may need to tune eps and min_samples
    LRFM['DBSCANClusters'] = dbscan.fit_predict(Z)

   
    # RFM[2] = RFM[2][RFM[2]['DBSCANClusters'] != -1]
    # s = set(LRFM[2]['LRFM'])
    # s

    # # Display the resulting DataFrame
    LRFM

    # Assuming the DBSCANClusters values range from 0 to 19

    cluster_labels = {
        19: "Long-Time Frequent Buyers",
        18: "Balanced Loyalists",
        17: "Cost-Conscious Product Buyers",
        16: "Occasional Buyers",
        15: "Continuously Moderate Spenders",
        14: "Currently Lost High-Value Customers",
        13: "Offer loving Cost-Conscious Veterans",
        12: "Less Interested Consumers",
        11: "Moderately Involved High Spending Veterans",
        10: "Irregular yet Frequent VIPs",
        9: "Cost-Conscious Constant Buyers",
        8: "Losing Long-Time High Spenders",
        7: "Momentary but Multiple Products Buyers",
        6: "Deal Seeking Customers",
        5: "Emerging Cost-Conscious Loyalists",
        4: "Semi-Premium Customers",
        3: "Elite/Perfect Customers",
        2: "Weekly Overspending Veterans",
        1: "Irregular & Moderate Buyers",
        0: "Least Interested Customers"
    }

    # Assign labels based on DBSCANClusters values
    LRFM['Conclusion_DBSCAN'] = LRFM['DBSCANClusters'].map(cluster_labels)
    LRFM

    l = ['L', 'F', 'M']

    for col in l:
        conditions = [
            LRFM[col].eq(1),
            LRFM[col].eq(2)
        ]
        choices = ['Low', 'Medium']

        LRFM[f'Status{col}'] = np.select(conditions, choices, default='High')
    LRFM

    conditions = [
        LRFM['R'] == 1,
        LRFM['R'] == 2
    ]

    choices = ['High', 'Medium']

    LRFM['StatusR'] = np.select(conditions, choices, default='Low')
    return LRFM

def plot_segmentation(LRFM):
        # Assuming LRFM[2] contains your data
    p = ggplot(LRFM, aes('Frequency', 'Monetary', group='Conclusion_DBSCAN')) \
        + geom_line() \
        + geom_point() \
        + facet_wrap('Conclusion_DBSCAN') \
        + scale_x_continuous() \
        + theme(figure_size=(15, 10))  # Adjust the figure size as needed

    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    p.save(buffer, format='png')
    buffer.seek(0)
    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()
    # Close the buffer
    buffer.close()
    # Return the base64 encoded image string
    return img_str



def lineplot_function(LRFM):
    # Order 'Conclusion_DBSCAN' based on mean 'Monetary' value
    order = LRFM.groupby('Conclusion_DBSCAN')['Monetary'].mean().sort_values().index

    # Sort the DataFrame based on the order
    sorted_df = LRFM.set_index('Conclusion_DBSCAN').loc[order].reset_index()

    # Set the size of the plot
    plt.figure(figsize=(16, 16))

    # Line chart for 'Monetary' by 'Conclusion_DBSCAN'
    sns.lineplot(x='Conclusion_DBSCAN', y='Monetary', data=sorted_df, label='Monetary')

    # Set plot title and labels
    plt.title('Line Chart of Monetary Factor as per Segmented Customers\n')
    plt.xlabel('Segmented Customers')
    plt.ylabel('Monetary Factors')

    # Show the legend
    plt.legend()

    # Rotate x-axis labels to the right by 45 degrees
    plt.xticks(rotation=30, ha='right')

    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()
    # Close the buffer
    buffer.close()
    # Return the base64 encoded image string
    return img_str


def treemap_visualization(LRFM):
    hierarchical_data = LRFM.reset_index().groupby(['Conclusion_DBSCAN', '_CustomerID']).agg({
        'Recency': 'sum',
        'Length': 'mean',
        'Frequency': 'first',
        'Monetary': 'first',
        'StatusR': 'first',
        'StatusF': 'first',
        'StatusM': 'first',
        'StatusL': 'first',
    }).reset_index()

    # Create the tree map
    fig = px.treemap(
        hierarchical_data,
        path=['Conclusion_DBSCAN', '_CustomerID'],
        values='Recency',
        color='Length',
        color_continuous_scale='viridis',
        title='Tree Map of Customer Segmentation',
        hover_data={
            '_CustomerID': True,
            'Recency': True,
            'Length': True,
            'Frequency': True,
            'Monetary': True,
            'StatusR': True,
            'StatusF': True,
            'StatusM': True,
            'StatusL': True,
            'Conclusion_DBSCAN': True
        }
    )

    # Pass the plot as HTML to the template
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

def dashboard(request):
    sales_data = import_dataset()
    fig_sales_data = plot_sales_data(sales_data)
    histogram = histogram_make(sales_data)
    pie = pie_chart(sales_data)
    LRFM = customer_segmentation(sales_data)
    fig_segmentation = plot_segmentation(LRFM)
    lineplot = lineplot_function(LRFM)
    treemap = treemap_visualization(LRFM)
    revenuetrend = revenue_trend(sales_data)
    orderquantity = order_quantity_over_time(sales_data)

    return render(request, 'segmentation_app/segmentation.html', {'fig_sales_data': fig_sales_data,
                                                                  'lineplot' : lineplot,
                                                                  'histogram' : histogram,
                                                                  'pie': pie,
                                                                  'revenuetrend': revenuetrend,
                                                                  'orderquantity': orderquantity,
                                                                  'fig_segmentation': fig_segmentation,
                                                                  'treemap': treemap})




# def dashboard(request):
#     data_type = {
#         '_CustomerID': str
#     }

#     sales_data = pd.read_csv("C:/Users/Dell/Downloads/sales_data.csv", dtype=data_type, parse_dates=['OrderDate'])
#     sales_data

#     sales_data['_CustomerID'].unique()

#     sales_data.head()

#     sales_data['Revenue'] = (sales_data['Unit Price'] - (sales_data['Unit Price'] * sales_data['Discount Applied']) -
#                              sales_data['Unit Cost']) * sales_data['Order Quantity']

#     # Calculate mean revenue for each customer
#     mean_revenue = sales_data.groupby('_CustomerID')['Revenue'].transform('mean')

#     # Create a new column 'MeanRevenue' in cutoff_in DataFrame
#     sales_data['MeanRevenue'] = mean_revenue

#     sales_data1 = (
#         sales_data
#         .reset_index()
#         .set_index('OrderDate')
#         [['Revenue']]
#         .resample('MS')
#         .sum()
#     )

#     sales_data1.plot()

#     df = sales_data.copy()
#     df['OrderDate'].isnull().sum()

#     df.dtypes

#     columns = ['OrderNumber', '_CustomerID', 'OrderDate', 'Revenue', 'MeanRevenue']
#     df_dataset = df[columns]

#     df_dataset

#     n_days = 180
#     max_date = df_dataset['OrderDate'].max()
#     cutoff = max_date - pd.to_timedelta(n_days, unit="d")
#     cutoff1 = cutoff - pd.to_timedelta(n_days, unit="d")

#     cutoff_in = df_dataset[df_dataset['OrderDate'] <= cutoff].copy()
#     cutoff_out = df_dataset[df_dataset['OrderDate'] > cutoff]
#     cutoff_in1 = df_dataset[(df_dataset['OrderDate'] <= cutoff) & (df_dataset['OrderDate'] > cutoff1)]

#     # Display cutoff_in DataFrame
#     cutoff_in1

#     # Assuming targets_df is defined
#     targets_df = cutoff_out.groupby('_CustomerID')['Revenue'].sum().reset_index()\
#         .rename(columns={'Revenue': 'Spent_last_30'})\
#         .assign(Spent_last_30_flag=1)

#     # Set '_CustomerID' as the index
#     targets_df.set_index('_CustomerID', inplace=True)

#     # Display targets_df
#     targets_df

#     features_df = cutoff_in.groupby('_CustomerID').agg({
#         'OrderDate': lambda v: (cutoff_in['OrderDate'].max() + pd.Timedelta(days=1) - v.max()).days,
#         'OrderNumber': 'count',
#         'Revenue': 'sum',
#         'MeanRevenue': 'first'
#     })
#     features_df1 = cutoff_in1.groupby('_CustomerID').agg({
#         'OrderDate': lambda v: (cutoff_in['OrderDate'].max() + pd.Timedelta(days=1) - v.max()).days,
#         'OrderNumber': 'count',
#         'Revenue': 'sum',
#         'MeanRevenue': 'first'
#     })
#     features_df1

#     features_df.rename(
#         columns={
#             'OrderDate': 'Recency_before_last_30_days',
#             'OrderNumber': 'Frequency_before_last_30_days',
#             'Revenue': 'Monetary_before_last_30_days',
#             'MeanRevenue': 'Average_Money_before_last_30_days'
#         },
#         inplace=True
#     )
#     features_df1.rename(
#         columns={
#             'OrderDate': 'Recency_before_180_days_of_cutoff',
#             'OrderNumber': 'Frequency_before_180_days_of_cutoff',
#             'Revenue': 'Monetary_before_180_days_of_cutoff',
#             'MeanRevenue': 'Average_Money_before_180_days_of_cutoff'
#         },
#         inplace=True
#     )
#     features_df1

#     features_df = features_df.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)
#     features_df1 = features_df1.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)

#     features_df

#     df_dataset['LengthOfUsage'] = (df_dataset.groupby('_CustomerID')['OrderDate'].transform('max') -
#                                    df_dataset.groupby('_CustomerID')['OrderDate'].transform('min')).dt.days.copy()

#     today_date = pd.to_datetime('2021-01-01')

#     lrfm_dataset = df_dataset.groupby('_CustomerID').agg({
#         'OrderDate': lambda v: (today_date - v.max()).days,
#         'OrderNumber': 'count',
#         'Revenue': 'sum',
#         'LengthOfUsage': 'first'  # Length of Usage
#     })
#     lrfm_dataset

#     lrfm_dataset.rename(
#         columns={
#             'OrderDate': 'Recency',
#             'OrderNumber': 'Frequency',
#             'Revenue': 'Monetary',
#             'LengthOfUsage': 'Length'
#         },
#         inplace=True
#     )
#     lrfm_dataset

#     lrfm_dataset['R'] = np.where(lrfm_dataset['Recency'] < 4.001, 1,
#                                  np.where((4.001 <= lrfm_dataset['Recency']) & (lrfm_dataset['Recency'] <= 8.000), 2, 3))

#     lrfm_dataset['F'] = np.where(lrfm_dataset['Frequency'] < 152.333, 1,
#                                  np.where((152.333 <= lrfm_dataset['Frequency']) & (lrfm_dataset['Frequency'] <= 164.00),
#                                           2, 3))

#     lrfm_dataset['M'] = np.where(lrfm_dataset['Monetary'] < 402127.000, 1,
#                                  np.where((402127.000 <= lrfm_dataset['Monetary']) & (lrfm_dataset['Monetary'] <= 443110),
#                                           2, 3))

#     lrfm_dataset['L'] = np.where(lrfm_dataset['Length'] < 200.00, 1,
#                                  np.where((500.00 <= lrfm_dataset['Length']) & (lrfm_dataset['Length'] <= 900.00), 2,
#                                           3))

#     lrfm = lrfm_dataset
#     lrfm

#     lrfm['lrfm_group'] = lrfm[['L', 'R', 'F', 'M']].apply(lambda v: '-'.join(v.astype(str)), axis=1)

#     lrfm[['L', 'R', 'F', 'M']] = lrfm[['L', 'R', 'F', 'M']]
#     lrfm['lrfm_score_total'] = lrfm[['L', 'R', 'F', 'M']].sum(axis=1)

#     LRFM = lrfm.copy()

#     LRFM = LRFM.drop(columns=['lrfm_group'])

#     LRFM['lrfm'] = LRFM['R'] * 1000 + LRFM['F'] * 100 + LRFM['M'] * 10 + LRFM['L']

#     LRFM

#     Z = LRFM[['L', 'R', 'F', 'M']]

#     # Choose the number of clusters (you can tune this parameter)
#     n_clusters = 30

#     # Apply DBSCAN clustering
#     dbscan = DBSCAN(eps=0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, min_samples=1)  # You may need to tune eps and min_samples
#     LRFM['DBSCANClusters'] = dbscan.fit_predict(Z)

   
#     # RFM[2] = RFM[2][RFM[2]['DBSCANClusters'] != -1]
#     # s = set(LRFM[2]['LRFM'])
#     # s

#     # # Display the resulting DataFrame
#     LRFM

#     # Assuming the DBSCANClusters values range from 0 to 19

#     cluster_labels = {
#         19: "Long-Time Frequent Buyers",
#         18: "Balanced Loyalists",
#         17: "Cost-Conscious Product Buyers",
#         16: "Occasional Buyers",
#         15: "Continuously Moderate Spenders",
#         14: "Currently Lost High-Value Customers",
#         13: "Offer loving Cost-Conscious Veterans",
#         12: "Less Interested Consumers",
#         11: "Moderately Involved High Spending Veterans",
#         10: "Irregular yet Frequent VIPs",
#         9: "Cost-Conscious Constant Buyers",
#         8: "Losing Long-Time High Spenders",
#         7: "Momentary but Multiple Products Buyers",
#         6: "Deal Seeking Customers",
#         5: "Emerging Cost-Conscious Loyalists",
#         4: "Semi-Premium Customers",
#         3: "Elite/Perfect Customers",
#         2: "Weekly Overspending Veterans",
#         1: "Irregular & Moderate Buyers",
#         0: "Least Interested Customers"
#     }

#     # Assign labels based on DBSCANClusters values
#     LRFM['Conclusion_DBSCAN'] = LRFM['DBSCANClusters'].map(cluster_labels)
#     LRFM

#     l = ['L', 'F', 'M']

#     for col in l:
#         conditions = [
#             LRFM[col].eq(1),
#             LRFM[col].eq(2)
#         ]
#         choices = ['Low', 'Medium']

#         LRFM[f'Status{col}'] = np.select(conditions, choices, default='High')
#     LRFM

#     conditions = [
#         LRFM['R'] == 1,
#         LRFM['R'] == 2
#     ]

#     choices = ['High', 'Medium']

#     LRFM['StatusR'] = np.select(conditions, choices, default='Low')
#     LRFM

#     # Assuming LRFM[2] contains your data
#     p = ggplot(LRFM, aes('Frequency', 'Monetary', group='Conclusion_DBSCAN')) \
#         + geom_line() \
#         + geom_point() \
#         + facet_wrap('Conclusion_DBSCAN') \
#         + scale_x_continuous() \
#         + theme(figure_size=(15, 10))  # Adjust the figure size as needed

#     print(p)

#     # Order 'Conclusion_DBSCAN' based on mean 'Monetary' value
#     order = LRFM.groupby('Conclusion_DBSCAN')['Monetary'].mean().sort_values().index

#     # Sort the DataFrame based on the order
#     sorted_df = LRFM.set_index('Conclusion_DBSCAN').loc[order].reset_index()

#     # Set the size of the plot
#     plt.figure(figsize=(16, 16))

#     # Line chart for 'Monetary' by 'Conclusion_DBSCAN'
#     sns.lineplot(x='Conclusion_DBSCAN', y='Monetary', data=sorted_df, label='Monetary')

#     # Set plot title and labels
#     plt.title('Line Chart of Monetary Factor as per Segmented Customers\n')
#     plt.xlabel('Segmented Customers')
#     plt.ylabel('Monetary Factors')

#     # Show the legend
#     plt.legend()

#     # Rotate x-axis labels to the right by 45 degrees
#     plt.xticks(rotation=30, ha='right')

#     # Show the plot
#     plt.show()

#     hierarchical_data = LRFM.reset_index().groupby(['Conclusion_DBSCAN', '_CustomerID']).agg({
#         'Recency': 'sum',
#         'Length': 'mean',
#         'Frequency': 'first',
#         'Monetary': 'first',
#         'StatusR': 'first',
#         'StatusF': 'first',
#         'StatusM': 'first',
#         'StatusL': 'first',
#     }).reset_index()

#     # Create the tree map
#     fig = px.treemap(
#         hierarchical_data,
#         path=['Conclusion_DBSCAN', '_CustomerID'],
#         values='Recency',
#         color='Length',
#         color_continuous_scale='viridis',
#         title='Tree Map of Customer Segmentation',
#         hover_data={
#             '_CustomerID': True,
#             'Recency': True,
#             'Length': True,
#             'Frequency': True,
#             'Monetary': True,
#             'StatusR': True,
#             'StatusF': True,
#             'StatusM': True,
#             'StatusL': True,
#             'Conclusion_DBSCAN': True
#         }
#     )

# # Pass the plot as HTML to the template
#     plot_div = plot(fig, output_type='div', include_plotlyjs=False)

#     return render(request, 'segmentation_app/segmentation.html', {'plot_div': plot_div})

