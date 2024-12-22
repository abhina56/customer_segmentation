from django.shortcuts import render, redirect
from plotly.offline import plot
import plotly.express as px
import io
import base64
from PIL import Image
from django.contrib import messages
from django.urls import reverse
import hashlib  # Import hashlib library for password encryption
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
import MySQLdb
from django.db import connection
import json
from django.http import JsonResponse
from .models import User  # Import your User model
from django.db import IntegrityError
import bcrypt
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import re
import random
from django.core.mail import send_mail
import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import cross_val_predict
from matplotlib.colors import ListedColormap










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

    columns = ['OrderNumber', '_CustomerID', 'OrderDate', 'Revenue', 'MeanRevenue']
    df_dataset = df[columns]


    n_days = 180
    max_date = df_dataset['OrderDate'].max()
    cutoff = max_date - pd.to_timedelta(n_days, unit="d")
    cutoff1 = cutoff - pd.to_timedelta(n_days, unit="d")

    cutoff_in = df_dataset[df_dataset['OrderDate'] <= cutoff].copy()
    cutoff_out = df_dataset[df_dataset['OrderDate'] > cutoff]
    cutoff_in1 = df_dataset[(df_dataset['OrderDate'] <= cutoff) & (df_dataset['OrderDate'] > cutoff1)]


    # Assuming targets_df is defined
    targets_df = cutoff_out.groupby('_CustomerID')['Revenue'].sum().reset_index()\
        .rename(columns={'Revenue': 'Spent_last_180'})\
        .assign(Spent_last_180_flag=1)

    # Set '_CustomerID' as the index
    targets_df.set_index('_CustomerID', inplace=True)

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

    features_df = features_df.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)
    features_df1 = features_df1.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)


    df_dataset['LengthOfUsage'] = (df_dataset.groupby('_CustomerID')['OrderDate'].transform('max') -
                                   df_dataset.groupby('_CustomerID')['OrderDate'].transform('min')).dt.days.copy()

    today_date = pd.to_datetime('2021-01-01')

    lrfm_dataset = df_dataset.groupby('_CustomerID').agg({
        'OrderDate': lambda v: (today_date - v.max()).days,
        'OrderNumber': 'count',
        'Revenue': 'sum',
        'LengthOfUsage': 'first'  # Length of Usage
    })

    lrfm_dataset.rename(
        columns={
            'OrderDate': 'Recency',
            'OrderNumber': 'Frequency',
            'Revenue': 'Monetary',
            'LengthOfUsage': 'Length'
        },
        inplace=True
    )

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

    lrfm['lrfm_group'] = lrfm[['L', 'R', 'F', 'M']].apply(lambda v: '-'.join(v.astype(str)), axis=1)

    lrfm[['L', 'R', 'F', 'M']] = lrfm[['L', 'R', 'F', 'M']]
    lrfm['lrfm_score_total'] = lrfm[['L', 'R', 'F', 'M']].sum(axis=1)

    LRFM = lrfm.copy()

    LRFM = LRFM.drop(columns=['lrfm_group'])

    LRFM['lrfm'] = LRFM['R'] * 1000 + LRFM['F'] * 100 + LRFM['M'] * 10 + LRFM['L']

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

    # Assuming the DBSCANClusters values range from 0 to 19

    cluster_labels = {
        19: "Currently Lost High-Value Customers",
        18: "Momentary but Multiple Products Buyers",
        17: "Irregular yet Frequent VIPs",
        16: "Irregular & Moderate Buyers",
        15: "Less Interested Consumers",
        14: "Occasional Buyers",
        13: "Least Interested Customers",
        12: "Weekly Overspending Veterans",
        11: "Long-Time Frequent Buyers",
        10: "Moderately Involved High Spending Veterans",
        9: "Balanced Loyalists",
        8: "Emerging Cost-Conscious Loyalists",
        7: "Offer loving Cost-Conscious Veterans",
        6: "Elite/Perfect Customers",
        5: "Semi-Premium Customers",
        4: "Losing Long-Time High Spenders",
        3: "Continuously Moderate Spenders",
        2: "Cost-Conscious Product Buyers",
        1: "Deal Seeking Customers",
        0: "Cost-Conscious Constant Buyers"
    }
    
    cluster_labels1 = {
        2323: "Long-Time Frequent Buyers",
        2223: "Balanced Loyalists",
        1213: "Cost-Conscious Product Buyers",
        3123: "Occasional Buyers",
        1223: "Continuously Moderate Spenders",
        3333: "Currently Lost High-Value Customers",
        2113: "Offer loving Cost-Conscious Veterans",
        3213: "Less Interested Consumers",
        2233: "Moderately Involved High Spending Veterans",
        3233: "Irregular yet Frequent VIPs",
        1113: "Cost-Conscious Constant Buyers",
        1233: "Losing Long-Time High Spenders",
        3323: "Momentary but Multiple Products Buyers",
        1123: "Deal Seeking Customers",
        2213: "Emerging Cost-Conscious Loyalists",
        1323: "Semi-Premium Customers",
        1333: "Elite/Perfect Customers",
        2333: "Weekly Overspending Veterans",
        3223: "Irregular & Moderate Buyers",
        3113: "Least Interested Customers"
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

    conditions = [
        LRFM['R'] == 1,
        LRFM['R'] == 2
    ]

    choices = ['High', 'Medium']

    LRFM['StatusR'] = np.select(conditions, choices, default='Low')
    return LRFM
    
def prediction(sales_data):
    df = sales_data.copy()

    columns = ['OrderNumber', '_CustomerID', 'OrderDate', 'Revenue', 'MeanRevenue']
    df_dataset = df[columns]

    n_days = 180
    max_date = df_dataset['OrderDate'].max()
    cutoff = max_date - pd.to_timedelta(n_days, unit="d")
    cutoff1 = cutoff - pd.to_timedelta(n_days, unit="d")

    cutoff_in = df_dataset[df_dataset['OrderDate'] <= cutoff].copy()
    cutoff_out = df_dataset[df_dataset['OrderDate'] > cutoff]
    cutoff_in1 = df_dataset[(df_dataset['OrderDate'] <= cutoff) & (df_dataset['OrderDate'] > cutoff1)]

    # Assuming targets_df is defined
    targets_df = cutoff_out.groupby('_CustomerID')['Revenue'].sum().reset_index()\
        .rename(columns={'Revenue': 'Spent_last_180'})\
        .assign(Spent_last_180_flag=1)

    # Set '_CustomerID' as the index
    targets_df.set_index('_CustomerID', inplace=True)

    features_df1 = cutoff_in1.groupby('_CustomerID').agg({
        'OrderDate': lambda v: (cutoff_in['OrderDate'].max() + pd.Timedelta(days=1) - v.max()).days,
        'OrderNumber': 'count',
        'Revenue': 'sum',
        'MeanRevenue': 'first'
    })

    features_df1.rename(
        columns={
            'OrderDate': 'Recency_before_180_days_of_cutoff',
            'OrderNumber': 'Frequency_before_180_days_of_cutoff',
            'Revenue': 'Monetary_before_180_days_of_cutoff',
            'MeanRevenue': 'Average_Money_before_180_days_of_cutoff'
        },
        inplace=True
    )

    features_df1 = features_df1.merge(targets_df, left_index=True, right_index=True, how='left').fillna(0)
    return features_df1




def features_df(features_df1,LRFM):
    X = features_df1[['Recency_before_180_days_of_cutoff', 'Frequency_before_180_days_of_cutoff', 'Monetary_before_180_days_of_cutoff', 'Average_Money_before_180_days_of_cutoff']]

    y_spend = features_df1['Spent_last_180']
    
    # Creating the Linear Regression model
    ols_model = LinearRegression()
    
    # Performing cross-validated predictions
    predicted_values = cross_val_predict(ols_model, X, y_spend, cv=5)
    
    # Displaying the predicted values
    print('Predicted Values:')
    print(predicted_values)

    predictions_df10 = pd.concat(
    [ 
        pd.DataFrame(predicted_values).set_axis(['pred_spend'], axis = 1),
    #         pd.Dataframe(predictions_clf)[[1]].set_axis([‘pred_prob’], axis = 1),
        features_df1.reset_index()
    ],
    axis = 1
    )  
    predictions_df10['_CustomerID1'] = predictions_df10['_CustomerID'].astype(int)
    
    # Sort the DataFrame by '_CustomerID'
    predictions_df10 = predictions_df10.sort_values('_CustomerID1')
    predictions_df10['spend_dif']=predictions_df10['Spent_last_180']-predictions_df10['pred_spend']
    predictions_df10['S'] = np.where(predictions_df10['spend_dif'] > 20000, 3, np.where((-2000 <= predictions_df10['spend_dif']) & (predictions_df10['spend_dif'] <= 20000), 2, 1))
    predictions_df10 = predictions_df10.merge(LRFM[['M', 'Conclusion_DBSCAN', 'StatusM']], left_on='_CustomerID', right_on='_CustomerID', how='left').fillna(0)
    predictions_df10['marketing_label'] = predictions_df10['S']*10+predictions_df10['M']
    predictions_df10['Spending_Status'] = np.where(predictions_df10['S'] == 1, "Low", np.where(predictions_df10['S'] == 2, "Medium", "High"))
    predictions_df10 = predictions_df10.sort_values(by='marketing_label', ascending=True)
    Z = predictions_df10[['S','M']]
    n_clusters = 30
    dbscan = DBSCAN(eps=0.1, min_samples=1)  # You may need to tune eps and min_samples
    predictions_df10['DBSCAN_marketing'] = dbscan.fit_predict(Z)
    cluster_labels = {
        8: "Create Elite Membership Program, Curating Personalized Luxury Travel Experiences For High-Spending Customers, Personalized Messages Appreciating Their Acquisitions, VIP Virtual Styling Studios For Designing Personalized Products, , Implement AI-Powered Styling Algorithms To Analyze Preferences According To Purchase Behaviour, Providing Private Shopping Consultations With Dedicated Stylists, Offering Art Investment Advisory Services For Customers Interested In Acquiring High-End Artworks ",
        7: "Creating Exclusive Rewards Program, Offer Personalized Luxury Experiences, Introduction To Limited-Edition Collectibles, Thank You Notes On Cross Platforms, Use Their Recognition For Public Advertisement, Implement AI-Powered Styling Algorithms To Analyze Customers, Organizing Exclusive Wellness Retreats, Conduction Of Youth-Driven Recreational Programs With Massive Publicity Reach Out To Assist With Product Selection,  Suggest Upgraded Products Based On Their Recent Purchases, Augmented Reality (AR) Shopping Experience",
        6: "Surprise And Delight Rewards Acknowledging Their Recent Involvement, Encouraging To Share Recent Purchases Using Branded Hashtags, Adding to Exclusive Online Communities Or Forums, Cross_Messaging Thank You Notes, Incentivizing To Refer Friends And Family By Offering Referral Rewards, Feature Them In Cross Platform Ads, Collaborate With Popular Influencers/Artists/Designers To Create Limited-Edition Product Collections, Allow Customized Design FOr Their Future Purchases, Implement Dynamic Pricing Strategies For Them, Offer Personalized Subscription Boxes",
        5: "Exclusive VIP Experiences Tailored To Their Interests, Creating Personalized Product Bundles, AR-Powered Concierge Services To Allow VIPs To Virtually Interact With Knowledgeable Staff,  VIP Virtual Styling Studios For Designing Personalized Products, Early Access To Limited-Edition Products, Membership-Based Luxury Clubs Or Loyalty Programs As Appreciation, Invitation To Unboxing Events And Product Design Campaigns,  Reach Out To Assist With Product Selection, Using Their Testemonials Or Reviews to Pull Depreciating Customers",
        4: "Showing Gratitude By Thank-You Notifications And Notes, Highlighting Loyalty and Contributions To Brand's Success, Loyalty Rewards Or Points, Exclusive Benefits Or Perks, Using Data Analytics To Provide Personalized Product Recommendations, Suggest Upgraded Products Based On Their Recent Purchases, Engaging via events/challenges, Using Their Feedbacks For Other Customers, Building Groups in Socials Fpr Interaction, Use of Virtual Shopping Assistant Bots,  Reach Out To Assist With Product Selection",
        3: "Acknowledge Constant Purchasing Activity By Appreciation, Loyalty Rewards/Cashbacks, Personalized Product Recommendations, Targeted Upselling Techniques Using Cross-Platforms, Free Tickets As Gifts Of Some Recreational Events, Priority Shipping Or Dedicated Customer Support, Engaging Challenges Or Contests Involving Marketing, Keep Recommending Everything That Was Recommended Of Late, Using Their Testemonials Or Reviews to Pull Depreciating Customers, Guides Related To Their Recent Purchases, Creating A Group Of Similar People In Socials",
        2: "Personalized Messages Appreciating Their Past Acquisitions, Sharing Memories Of Their Previous Purchaes, VIP Treatment By Offering Exclusive Perks, Escalated VIP Incentives, Recommend High-Priced And Previous Purchased Items With Offers, Using Advanced Recommendation Algorithms, Notifying About Upgraded Items, Personalized Customer Service Experiences Such As Dedicated Account Managers, Reach Out To Assist With Product Selection, Organizing VIP Sales Or Invitation-Only Previews, Inviting to Product Unboxing Events, Surprise With Unexpected Gifts, Create A Group Of Similar People In Socials",
        1: "Leveraging Data Analytics To Provide Personalized Product Recommendations, Targeted Message/Notifications and Call Marketing, Reactivation Campaigns, Urgency Tactics Such As Limited-Time Discounts, Interactive Surveys, Cashback Offers, Grand Marketing Based On Previous Purchase Behaviour, Customer Appreciation Campaigns, Conduction Of Youth-Driven Recreational Programs With Massive Publicity, Cross-Channel Marketing Strategies, Experimentation WIth Age Group Based Interesting Notifcation Message, Creating A Group Of Similar People In Socials",
        0: "Targeted Message and Call Marketing, Retargeting Ads On Various Platforms, Loyalty Rewards/Cashback/Points, Special Perks Such As Home-To-Home Marketing, Exclusive Offers On Relatively Less-Priced Products, Limited-Time Grand Promotions, Guides/Tutorials/Tips Regarding Such Segments' Previous Purchases, Sharing Customer Rankings/Reviews Publicly, Interactive Surveys For Engagement, Recommending Complementary Products, Offering Bundle Deals, Building a Community Among Customers, Adjustment of Strategies, Augmented Reality (AR) Shopping Experience, Use of Virtual Shopping Assistant Bot"
    }
    predictions_df10['Marketing_Strategies'] = predictions_df10['DBSCAN_marketing'].map(cluster_labels)


    return predictions_df10

def print_conclusions_dbscan_by_label(df):
    def get_conclusions_dbscan_for_labels(df, labels):
        conclusions_by_label = {}
        for label in labels:
            label_data = df[df['marketing_label'] == label]
            grouped_data = label_data.groupby('Conclusion_DBSCAN')['_CustomerID'].apply(list).reset_index()
            conclusions = grouped_data['Conclusion_DBSCAN'].tolist()
            customer_ids = grouped_data['_CustomerID'].apply(lambda x: ', '.join(map(str, x))).tolist()
            spending_status = label_data['Spending_Status'].iloc[0]  # Get the Spending_Status for the label
            status_m = label_data['StatusM'].iloc[0]  # Get the StatusM for the label
            conclusions_by_label[label] = {'conclusions': conclusions, 'customer_ids': customer_ids,
                                            'spending_status': spending_status, 'status_m': status_m}
        return conclusions_by_label

    marketing_labels_of_interest = df['marketing_label'].unique()
    conclusion_dbscan_by_label = get_conclusions_dbscan_for_labels(df, marketing_labels_of_interest)

    # Get the first marketing strategy for each label
    marketing_strategies_by_label = df.groupby('marketing_label')['Marketing_Strategies'].first().to_dict()

    output = ""

    for label, data in conclusion_dbscan_by_label.items():
        count = 1
        conclusions = data['conclusions']
        customer_ids = data['customer_ids']
        marketing_strategies = marketing_strategies_by_label[label].split(', ')  # Split the marketing strategies
        spending_status = data['spending_status']  # Retrieve spending status for the label
        status_m = data['status_m']  # Retrieve StatusM for the label
        output += f"Customers With {spending_status} Spending (Recent) & {status_m} Spending (Overall):\n\n"
        for conclusion, customer_id in zip(conclusions, customer_ids):
            output += f"  {conclusion} (Customer IDs: {customer_id})\n\n"
        for strategy in marketing_strategies:
            output += f"  Marketing Strategy {count}: {strategy}\n"
            count += 1
        output += "\n"

    return output


def barplot(predictions_df10):
    # Set the style for better visualization
    sns.set(style="whitegrid")

    # Create a bar chart
    fig = plt.figure(figsize=(25, 18))
    bar_width = 0.3
    bar_spacing = 0.1 # Set the desired spacing between bars

    # Set x-axis ticks to the sorted values of '_CustomerID'
    plt.xticks(range(len(predictions_df10['_CustomerID'])), predictions_df10['_CustomerID'])

    # Plot the 'Spent_last_30' on the x-axis with adjusted x-coordinates for spacing
    plt.bar(range(len(predictions_df10['_CustomerID'])), predictions_df10['Spent_last_180'], bar_width, label='Spent_last_180')

    # Plot the 'pred_spend' on the x-axis with adjusted x-coordinates for spacing
    plt.bar([i + (bar_width + bar_spacing) for i in range(len(predictions_df10['_CustomerID1']))], predictions_df10['pred_spend'], bar_width, label='pred_spend')

    # Set labels and title with increased font size
    plt.xlabel('Customer ID', fontsize=25)
    plt.ylabel('Spending Amount', fontsize=25)
    plt.title('Comparison between Spent Amount in last 180 days and Predicted/Expected for each Customer', fontsize=25)
    plt.xticks(rotation=90, fontsize=25)  # Rotate and increase font size of x-axis labels
    plt.yticks(fontsize=25)  # Increase font size of y-axis labels
    plt.legend(fontsize=12)

    # # Show the plot
    # plt.show()

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


def feature_importance(features_df1):
    X = features_df1[['Recency_before_180_days_of_cutoff', 'Frequency_before_180_days_of_cutoff', 'Monetary_before_180_days_of_cutoff', 'Average_Money_before_180_days_of_cutoff']]
    y_spend = features_df1['Spent_last_180']

    # Fitting the model
    ols_model = LinearRegression()
    ols_model.fit(X, y_spend)
    
    # Extract feature importances (coefficients)
    feature_importances10 = ols_model.coef_
    
    # Take the absolute values of feature importances
    abs_feature_importances10 = abs(feature_importances10)
    
    # Create a DataFrame to store feature names and their corresponding importance values
    feature_importance_df10 = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs_feature_importances10
    }).sort_values(by='Importance', ascending=False)
    
    # Manually assign labels to features
    feature_labels = {
        'Recency_before_180_days_of_cutoff': 'Recency',
        'Frequency_before_180_days_of_cutoff': 'Frequency',
        'Monetary_before_180_days_of_cutoff': 'Monetary',
        'Average_Money_before_180_days_of_cutoff': 'Average Money'
    }
    # Replace feature names with labels
    feature_importance_df10['Feature'] = feature_importance_df10['Feature'].map(feature_labels)
    
    # Create a ggplot visualization using Plotnine
    plot = (
        ggplot(feature_importance_df10, aes(x='Feature', y='Importance'))
        + geom_col()
        + coord_flip()
    )
    
    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    plot.save(buffer, format='png')
    buffer.seek(0)
    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()
    # Close the buffer
    buffer.close()
    # Return the base64 encoded image string
    return img_str


def no_of_segments(LRFM):
    s = set(LRFM['lrfm'])
    l = len(s)
    return l

def no_of_customers(LRFM):
    s = set(LRFM.index)
    l = len(s)
    return l

def segments_no(LRFM):
    s = set(LRFM['Conclusion_DBSCAN'])
    s_str = ', '.join(s)
    return s_str

    

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
    plt.title('Line Chart of Monetary Factor as per Segmented Customers\n', fontsize=20)
    plt.xlabel('Segmented Customers', fontsize=12)
    plt.ylabel('Monetary Factors', fontsize=12)

    # Show the legend
    plt.legend()

    # Rotate x-axis labels to the right by 45 degrees
    plt.xticks(rotation=20, ha='right', fontsize=12)

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
        },
        height=900, 
        width=1100
    )

    # Update layout to center the title and make it bold
    fig.update_layout(
        paper_bgcolor='black',  # Set paper background color to black (for the surrounding space)
        font_color='white'  # Set font color to white
    )

    # # Set font color for all text-based attributes
    # fig.update_traces(textfont_color='blue')  # Set text font color to red

    # Pass the plot as HTML to the template
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

def three_d_plot(LRFM):
    # Assuming you have 'Recency', 'Frequency', 'Monetary', 'Length', and 'Conclusion_DBSCAN' columns
    x = LRFM['Recency']
    y = LRFM['Frequency']
    z = LRFM['Monetary']

    # Create and fit a label encoder for 'Conclusion_DBSCAN'
    label_encoder = LabelEncoder()
    LRFM['DBSCANClusters'] = label_encoder.fit_transform(LRFM['Conclusion_DBSCAN'])

    # Create a custom color palette with 20 distinct colors
    custom_palette = sns.color_palette("tab20", 20)
    cmap = ListedColormap(custom_palette)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=LRFM['DBSCANClusters'], cmap=cmap, s=50)

    # Add labels
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')

    # Add 'Length' as a text annotation for each point above the plot
    for i, txt in enumerate(LRFM.index):
        ax.text(x[i], y[i], z[i], f'{LRFM["Length"].loc[txt]}', fontsize=8, ha='center', va='bottom')

    # Add a colorbar to show the mapping of colors to clusters
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(label_encoder.classes_)))
    cbar.set_label('Conclusion_DBSCAN')
    cbar.set_ticklabels(label_encoder.classes_)

    plt.title('3D Scatter Plot of Customer Segmentation with Length')

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

def bar_count_segments(LRFM):
    # Assuming your DBSCAN results are in a column named 'Conclusion_DBSCAN'
    plot=plt.figure(figsize=(17, 17))

    ax = sns.countplot(x='Conclusion_DBSCAN', data=LRFM, palette='viridis')
    plt.title('Segmented Customers and Quantity\n', fontsize=20)
    plt.xlabel('\nSegmented Customers', fontsize=13)
    # plt.ylabel('Total Customers', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    
    # Rotate x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    # plt.show()

    # Save the figure to a bytes buffer
    buffer = io.BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    # Encode the image bytes as base64
    img_str = base64.b64encode(buffer.read()).decode()
    # Close the buffer
    buffer.close()
    # Return the base64 encoded image string
    return img_str




def connect_to_mysql():
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,  # Default MySQL port
        user='',  # Your MySQL username
        passwd='',  # Your MySQL password
        db='',  # Your database name
        charset='utf8'
    )
    return conn


def login(request):
    # User.objects.all().delete()
    # with connection.cursor() as cursor:
    #     cursor.execute("ALTER TABLE segmentation_app_user AUTO_INCREMENT = 1")

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Query the database
        with connection.cursor() as cursor:
            try:
                cursor.execute("SELECT * FROM segmentation_app_user WHERE username = %s", (email,))
                user = cursor.fetchone()

                if user:
                    # Check if the password matches
                    if bcrypt.checkpw(password.encode(), user[2].encode()):  # Assuming password is at index 2
                        # User authenticated, perform login action (e.g., set session variable, etc.)
                        messages.success(request, 'Login successful!')
                        return redirect(reverse('segmentation_app:home'))  # Redirect to the home page after successful login
                    else:
                        # Incorrect password
                        messages.error(request, 'Invalid password.')
                else:
                    # User not found
                    messages.error(request, 'User does not exist.')
            except Exception as e:
                # Handle database errors
                messages.error(request, 'An error occurred while processing your request.')
                # Log the error for debugging
                print(e)

    return render(request, 'segmentation_app/login.html')


def create_new_user(request):
    # Parse form data
    first_name = request.POST.get('first_name')
    last_name = request.POST.get('last_name')
    phone_number = request.POST.get('phonenumber')
    purpose = request.POST.get('purpose')
    gender = request.POST.get('gender')
    province = request.POST.get('province')
    district = request.POST.get('district')
    dob = request.POST.get('dob')
    username = request.POST.get('username')
    password = request.POST.get('password')

    # Enforce password complexity
    if not re.match(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$", password):
        messages.error(request, "Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")
        return None

    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    name = f'{first_name} {last_name}'
    address = f'{province}, {district}'

    # Check if the email already exists
    with connection.cursor() as cursor:
        cursor.execute("SELECT username FROM segmentation_app_user WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            messages.error(request, "This email is already registered. Enter a new one.")
            return None

    # Generate a unique verification token
    verification_token = ''.join([str(random.randint(0, 9)) for _ in range(6)])

    # Create and return the new user object
    new_user = User(
        name=name,
        phone_number=phone_number,
        purpose=purpose,
        gender=gender,
        address=address,
        dob=dob,
        username=username,
        password=hashed_password.decode(),  # Decode the hashed password before saving
        verification_token=verification_token,  # Save the verification token
        Verified="False"  # Mark the user as unverified
    )
    return new_user


def signup(request):
    messages.error(request, "")

    if request.method == 'POST':
        new_user = create_new_user(request)
        if new_user:
            # Send verification email
            email_subject = "Verify Your Email Address"
            email_body = f"Dear {new_user.name},\n\nPlease paste the following code to verify your email address:\n{new_user.verification_token}\n\nRegards,\nThe Segmentation App Team"
            send_mail(email_subject, email_body, 'from@example.com', [new_user.username])

            # Store new_user object in session for use in verify function
            request.session['new_user'] = {
                'name': new_user.name,
                'phone_number': new_user.phone_number,
                'purpose': new_user.purpose,
                'gender': new_user.gender,
                'address': new_user.address,
                'dob': new_user.dob,
                'username': new_user.username,
                'password': new_user.password,
                'verification_token': new_user.verification_token,
                'Verified' : "False"
            }

           

            messages.success(request, 'An email has been sent to your email address. Please follow the instructions to verify your account.')
            return redirect('segmentation_app:verify')  # Replace 'success_page_url' with your actual success page URL

    # If the request method is not POST or new_user creation fails, render the signup form template
    return render(request, 'segmentation_app/signup.html')

def verify(request):
    if request.method == 'POST':
        verification_code = request.POST.get('verification_code')
        new_user_data = request.session.get('new_user')  # Retrieve new_user data from session
        if new_user_data:
            if new_user_data['verification_token'] == verification_code:
                # Create user object from session data
                new_user = User(
                    name=new_user_data['name'],
                    phone_number=new_user_data['phone_number'],
                    purpose=new_user_data['purpose'],
                    gender=new_user_data['gender'],
                    address=new_user_data['address'],
                    dob=new_user_data['dob'],
                    username=new_user_data['username'],
                    password=new_user_data['password'],
                    verification_token=verification_code,
                    Verified = "False"
                )
                    # Check if the email already exists
                with connection.cursor() as cursor:
                    cursor.execute("SELECT username FROM segmentation_app_user WHERE username = %s", (new_user_data['username'],))
                    existing_user = cursor.fetchone()
                    if existing_user:
                        messages.error(request, "This email is already registered. Enter a new one.")
                        return redirect('segmentation_app:signup')
               
                new_user.save()  # Save the user to the database
                messages.success(request, 'Your account has been verified!')
                return redirect('segmentation_app:payment_initiate')
            else:
                messages.error(request, 'Invalid verification code.')
                return redirect('segmentation_app:signup')

    return render(request, 'segmentation_app/loginverification.html')

def semipremium(request):
    # Call import_dataset function to get sales_data
    sales_data = import_dataset()

    # Call prediction function to get features_df1
    features_df1 = prediction(sales_data)
    LRFM = customer_segmentation(sales_data)
    treemap = treemap_visualization(LRFM)
    noofsegments = no_of_segments(LRFM)
    segments=segments_no(LRFM)
    customers=no_of_customers(LRFM)    
    data = plot_sales_data(sales_data)
    barcountsegments=bar_count_segments(LRFM)
    threedplot=three_d_plot(LRFM)
    lineplot = lineplot_function(LRFM)
    plotsegmentation = plot_segmentation(LRFM)


    # Call barplot function to get the base64 encoded image string
    predictions_df10=features_df(features_df1, LRFM)
    img_str = barplot(predictions_df10)
    featureimportance = feature_importance(features_df1)
    # Example usage:
    marketing_strategies = print_conclusions_dbscan_by_label(predictions_df10)


    # Pass the image string to the template
    return render(request, 'segmentation_app/semipremium.html', {'img_str': img_str,
                                                                 'treemap':treemap,
                                                                 'featureimportance':featureimportance,
                                                                 'noofsegments':noofsegments,
                                                                 'segments': segments,
                                                                 'data':data,
                                                                 'barcountsegments':barcountsegments,
                                                                 'threedplot':threedplot,
                                                                 'lineplot': lineplot,
                                                                 'plotsegmentation':plotsegmentation,
                                                                 'predictions_df10':predictions_df10,
                                                                 'customers':customers,
                                                                 'marketing_strategies': marketing_strategies})

# def payment_initiate(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         status = data.get('status')

#         if status == "Semi-Premium":
#             new_user_data = request.session.get('new_user')
#             if new_user_data:
#                 username = new_user_data.get('username')
#                 try:
#                     user = User.objects.get(username=username)
#                     user.Verified = "Semi-Premium"
#                     user.save()
#                     return JsonResponse({'message': 'User status updated successfully'})
#                 except User.DoesNotExist:
#                     return JsonResponse({'error': 'User does not exist'}, status=400)
#             else:
#                 return JsonResponse({'error': 'User data not found in session'}, status=400)
#         else:
#             return JsonResponse({'error': 'Invalid status'}, status=400)
    
#     # If it's not a POST request, render the payment page
#     new_user_data = request.session.get('new_user')
#     if new_user_data:
#         is_verified = new_user_data.get('Verified', False)
#         if is_verified:  # Check if user is verified
#             messages.success(request, 'Your account has been verified!')
#         else:
#             messages.error(request, 'User is not verified!')
#     else:
#         messages.error(request, 'User data not found in session!')

#     return render(request, 'segmentation_app/payment.html')

# def payment_initiate(request):
#     # Assuming you have stored some user data in the session
#     new_user_data = request.session.get('new_user')

#     try:
#         data = json.loads(request.body)
#     except json.JSONDecodeError:
#         return render(request, 'segmentation_app/payment.html')

#     if data is not None:
#         username = new_user_data.get('username')
#         try:
#             # Fetching the user from the database based on the username
#             user = User.objects.get(username=username)
            
#             # Assuming you want to update the user's Verified field with the productName received from Khalti
#             user.Verified = "Semi-Premium"
#             user.save()
#             messages.success(request, 'User status updated successfully!')
#             return render(request, 'segmentation_app/payment.html')  # Redirect to success URL after payment initiation
#         except User.DoesNotExist:
#             messages.error(request, 'User is not verified!')
#             return render(request, 'segmentation_app/payment.html')
    
    # return render(request, 'segmentation_app/payment.html')

async def payment_initiate(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            token = data.get('token')
            amount = data.get('amount')
            print(token, amount)  # Corrected logging statement

            payload = {
                'token': token,
                'amount': amount
            }

            headers = {
                'Authorization': 'Bearer ' + 'test_secret_key_cc1dc1c1b1a64875be35c81f4e7af92b'  # Corrected Authorization header
            }

            verification_response = await requests.post("https://khalti.com/api/v2/payment/verify/", json=payload, headers=headers)
            print(verification_response.json())  # Logging verification response

            return render(request, 'segmentation_app/payment.html')

        except Exception as e:
            print("Error occurred during payment verification:", str(e))
            return render(request, 'segmentation_app/payment.html')

    return render(request, 'segmentation_app/payment.html')

# def payment_initiate(request):
#     url = "https://a.khalti.com/api/v2/epayment/initiate/"

#     payload = {
#         "return_url": "http://example.com/",
#         "website_url": "https://example.com/",
#         "amount": "1000",   
#         "purchase_order_id": "Order01",
#         "purchase_order_name": "Semi-Premium",
#         "customer_info": {
#             "name": "Ram Bahadur",
#             "email": "test@khalti.com",
#             "phone": "9800000001"
#         }
#     }
#     headers = {
#         'Authorization': 'key test_secret_key_cc1dc1c1b1a64875be35c81f4e7af92b',
#         'Content-Type': 'application/json',
#     }

#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response_data = response.json()
#         print(JsonResponse(response_data, status=response.status_code))
#         return render(request, 'segmentation_app/payment.html')
#     except Exception as e:
#         return render(request, 'segmentation_app/payment.html')


def dashboard(request):
    sales_data = import_dataset()
    fig_sales_data = plot_sales_data(sales_data)
    histogram = histogram_make(sales_data)
    pie = pie_chart(sales_data)
    revenuetrend = revenue_trend(sales_data)
    orderquantity = order_quantity_over_time(sales_data)

    return render(request, 'segmentation_app/segmentation.html', {'fig_sales_data': fig_sales_data,
                                                                  'histogram' : histogram,
                                                                  'pie': pie,
                                                                  'revenuetrend': revenuetrend,
                                                                  'orderquantity': orderquantity
                                                                  })



def get_nepal_location(request):
    file_path = os.path.join(os.path.dirname(__file__), '{% static "images/SegmentationMap.png" %}')
    with open(file_path) as json_file:
        data = json.load(json_file)
    return JsonResponse(data)

def home(request):
    sales_data = import_dataset()
    LRFM = customer_segmentation(sales_data)
    data = plot_sales_data(sales_data)
    noofsegments = no_of_segments(LRFM)
    lineplot = lineplot_function(LRFM)
    plotsegmentation = plot_segmentation(LRFM)
    treemap = treemap_visualization(LRFM)
    segments = segments_no(LRFM)
    barcountsegments=bar_count_segments(LRFM)
    threedplot=three_d_plot(LRFM)


    return render(request, 'segmentation_app/segmentation2.html', {
        'noofsegments': noofsegments,
        'lineplot': lineplot,
        'treemap': treemap,
        'plotsegmentation': plotsegmentation,
        'segments': segments,
        'barcountsegments':barcountsegments,
        'data':data,
        'threedplot':threedplot,
        'LRFM': LRFM
    })





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

