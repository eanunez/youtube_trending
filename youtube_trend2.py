''' Author: Emmanuel Nunez
    Credit: Quan Nguyen
'''

import pandas as pd
import numpy as np

import json, sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


class PltAnalytics(object):

    @staticmethod
    def plt_corr(corr_obj):

        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        sns.heatmap(corr_obj, mask=np.zeros_like(corr_obj, dtype=np.bool),
                    cmap=sns.diverging_palette(240, 10, as_cmap=True), square=True, ax=ax1, annot=True)
        ax1.set_title('US-Video Correlation Matrix')
        ax1.set_xticklabels(corr_obj.columns.values, rotation=45, fontsize= 8)
        ax1.set_yticklabels(corr_obj.columns.values, rotation=45, fontsize= 8)

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(top=.9, bottom=0.2)

        plt.show()

    @staticmethod
    def visualize_most(my_df, column, num=10):  # getting the top 10 videos by default
        '''
        visualize most statistics such as most likes, dislikes, views, comments, etc.
        :param my_df: DataFrame object
        :param column: name of column of interest in type 'str'
        :param num: number of items to; default 10; type 'int'
        :return: plot
        '''
        sorted_df = my_df.sort_values(column, ascending=False).iloc[:num]

        ax = sorted_df[column].plot(kind='bar', figsize=(8, 6))

        # customizes the video titles, for asthetic purposes for the bar chart
        labels = []
        for item in sorted_df['title']:
            labels.append(item[:10] + '...')
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel('counts')
        ax.set_title('Most Number of ' + column.title(), fontsize=10)

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(top=.92, bottom=0.2)

        plt.show()

    @staticmethod
    def visualize_statistics(my_df, id_list):  # taking a list of video ids

        target_df = my_df.loc[id_list]

        ax = target_df[['views', 'likes', 'dislikes', 'comment_count']].plot(kind='bar', figsize=(8,6),
                                                    logy=True)

        # customizes the video titles, for asthetic purposes for the bar chart
        labels = []
        for item in target_df['title']:
            labels.append(item[:10] + '...')
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.set_ylabel('counts')

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(top=.92, bottom=0.2)

        plt.show()

    @staticmethod
    def visualize_like_dislike(my_df, id_list):
        target_df = my_df.loc[id_list]

        ax = target_df[['likes', 'dislikes']].plot(kind ='bar', stacked=True, figsize= (8, 6))

        # customizes the video titles, for asthetic purposes for the bar chart
        labels = []
        for item in target_df['title']:
            labels.append(item[:10] + '...')
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.set_ylabel('counts')

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(top=.92, bottom=0.2)

        plt.show()

    @staticmethod
    def visualize_category(category_count):

        ax = category_count.plot(kind='bar', figsize= (8, 6))

        ax.set_xticklabels(labels=category_count.index, rotation=45, fontsize=7)
        ax.set_ylabel('counts')
        ax.set_title('Category Visualization')

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(top=.92, bottom=0.2)

        plt.show()

def category_column():
    '''
    Here we are adding the category column after the category_id column, using the US_category_id.json file for lookup.
    :return: dictionary containing category id and category column
    '''
    # creates a dictionary that maps `category_id` to `category`
    id_to_category = {}

    with open('youtube_new\\US_category_id.json', 'r') as f:
        data = json.load(f)
        for category in data['items']:
            id_to_category[category['id']] = category['snippet']['title']

    return id_to_category

def main_process(argv):

    # Reading Dataset
    file_name = 'youtube_new\\USvideos.csv' # change this if you want to read a different dataset
    my_df = pd.read_csv(file_name, index_col='video_id')
    # print(my_df.head())

    # Change Trending dates to correct format
    my_df['trending_date'] = pd.to_datetime(my_df['trending_date'], format='%y.%d.%m')
    my_df['trending_date'].head()

    # Change Publish time to correct format
    my_df['publish_time'] = pd.to_datetime(my_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    my_df['publish_time'].head()


    # separates date and time into two columns from 'publish_time' column
    my_df.insert(4, 'publish_date', my_df['publish_time'].dt.date)
    my_df['publish_time'] = my_df['publish_time'].dt.time
    my_df[['publish_date', 'publish_time']].head()

    # Manually assign data type 'int' to columns instead of floats to save up memory space
    type_int_list = ['views', 'likes', 'dislikes', 'comment_count']
    for column in type_int_list:
        my_df[column] = my_df[column].astype(int)

    # Nominal type is 'str' for category id
    type_str_list = ['category_id']
    for column in type_str_list:
        my_df[column] = my_df[column].astype(str)

    # Insert category column
    id_to_category = category_column()
    my_df.insert(4, 'category', my_df['category_id'].map(id_to_category))
    my_df[['category_id', 'category']].head()

    # correlation analysis
    keep_columns = ['views', 'likes', 'dislikes',
                    'comment_count']  # only looking at correlations between these variables
    corr_matrix = my_df[keep_columns].corr()
    # print(corr_matrix)

    new_plt = PltAnalytics()
    new_plt.plt_corr(corr_matrix)

    # Drop duplicates-only keep the last video to find the latest updated stats
    print(my_df.shape)
    my_df = my_df[~my_df.index.duplicated(keep='last')]
    print(my_df.shape)
    my_df.index.duplicated().any()  # verify no duplicates

    # call visualizing function
    # ===== Comment and uncomment for type of plot =======
    # new_plt.visualize_most(my_df, 'views')
    # new_plt.visualize_most(my_df, 'likes')
    # new_plt.visualize_most(my_df, 'dislikes')
    # new_plt.visualize_most(my_df, 'comment_count')

    # Generate random sample of video_id to plot
    # sample_id_list = my_df.sample(n=10, random_state=4).index  # creates a random sample of 10 video IDs
    # new_plt.visualize_statistics(my_df, sample_id_list)
    # new_plt.visualize_like_dislike(my_df, sample_id_list)

    # Category Analysis
    # category_count = my_df['category'].value_counts()  # frequency for each category
    # new_plt.visualize_category(category_count)


if __name__ == '__main__':
    main_process(sys.argv)