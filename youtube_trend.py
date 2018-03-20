'''
Author: Emmanuel Nunez
Sample of Data Analytics on Youtube Trending Using Python. 
This simple script demonstrates the computing power of Python along with Pandas package
to manipulate enormous amount of data. The data contents several months of youtube trending videos, 
stored in both in .csv and .json file taken from
https://www.kaggle.com/donyoe/exploring-youtube-trending-statistics-eda/data
Copyright of Contributors: Skalskip and Donyoe of Kaggle.com
'''

import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
import seaborn as sns


class PlotTrend(object):

    @staticmethod
    def trend_corr(videos):

        column_names = ['category_id','views', 'likes', 'dislikes','comment_count']

        corr = videos.corr()
        selected_corr = corr.loc[column_names, column_names]
        print(selected_corr)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        sns.heatmap(selected_corr, mask=np.zeros_like(selected_corr, dtype=np.bool),
                    cmap=sns.diverging_palette(10, 240, as_cmap=True), xticklabels=selected_corr.columns.values,
                    yticklabels=selected_corr.columns.values, square=True, ax=ax1)
        ax1.set_title('US-Video Correlation Matrix')
        ax1.set_xticklabels(selected_corr.columns.values, rotation=45, fontsize=8)
        ax1.set_yticklabels(selected_corr.columns.values, fontsize=8)

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(top=.9, bottom=0.2)

        plt.show()

    @staticmethod
    def publish_htm(videos):

        # We choose the 10 most trending videos

        most_views_likes = videos.sort_values(by=['views', 'likes'], ascending=False).head(10)

        # Construction of HTML table with miniature photos assigned to the most popular movies
        table_content = ''
        max_title_length = 50

        for date, row in most_views_likes.T.iteritems():
            html_row = '<tr>'
            html_row += '<td><img src="' + str(row['thumbnail_link']) + '"style="width:100px;height:100px;"></td>'
            html_row += '<td>' + str(row['channel_title']) + '</td>'
            html_row += '<td>' + str(row['title']) + '</td>'
            html_row += '<td>' + str(row['views']) + '</td>'
            html_row += '<td>' + str(row['likes']) + '</td>'
            html_row += '<td>' + str(row['publish_date']) + '</td>'
            html_row += '<td>' + str(row['trending_date']) + '</td>'
            html_row += '<td>' + str(row['category']) + '</td>'

            table_content += html_row + '</tr>'

        html_str = '<table><tr><th>Photo</th><th>Channel Title</th>' \
                   '<th style="width:250px;">Title</th><th>Views</th>' \
                   '<th>Likes</th><th>Publish Date</th><th>Trending_Date</th>' \
                   '<th>Category</th></tr>{}</table>'.format(table_content)
        html_frame = '<html><head>Top 10 Most Views and Likes</head><body>' + html_str + '</body></html>'
        f = open("youtube_trending.html", "w", encoding='utf-8')
        f.write(html_frame)
        f.close()

    @staticmethod
    def publish_time(us_videos_first):
        # Initialization of the list storing counters for subsequent publication hours
        publish_h = [0] * 24

        for index, row in us_videos_first.iterrows():   # row iteration in pandas
            publish_h[row["publish_hour"]] += 1   # row item in 'publish hour' is matched to publish_h index, increments

        values = publish_h
        ind = np.arange(len(values))

        # Creating new plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.yaxis.grid()
        ax.xaxis.grid()
        bars = ax.bar(ind, values)

        # Sampling of Colormap
        '''for i, b in enumerate(bars):
            b.set_color(plt.cm.viridis((values[i] - min(values)) / (max(values) - min(values))))'''

        plt.ylabel('Number of videos that got trending', fontsize=20)
        plt.xlabel('Time of publishing', fontsize=20)
        plt.title('Best time to publish video', fontsize=35, fontweight='bold')
        plt.xticks(np.arange(0, len(ind), len(ind) / 6), [0, 4, 8, 12, 16, 20])

        plt.show()


def main(argv):
    # Load data
    us_videos = pd.read_csv('youtube_new\\USvideos.csv')
    us_videos_categories = pd.read_json('youtube_new\\US_category_id.json')

    # Quick check of data
    # print(us_videos.head(1))    # first line
    # print(us_videos.info())     # info of the head

    # Transforming Trending date column to datetime format
    us_videos['trending_date'] = pd.to_datetime(us_videos['trending_date'], format='%y.%d.%m').dt.date

    # Transforming Trending date column to datetime format and splitting into two separate ones
    publish_time = pd.to_datetime(us_videos['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    us_videos['publish_date'] = publish_time.dt.date
    us_videos['publish_time'] = publish_time.dt.time
    us_videos['publish_hour'] = publish_time.dt.hour

    # Check again for changes
    # print(us_videos.head(1))

    # We'll use a very nice python featur - dictionary comprehension, to extract most important data from US_category_id.json
    categories = {category['id']: category['snippet']['title'] for category in us_videos_categories['items']}

    # Now we will create new column that will represent name of category
    us_videos.insert(4, 'category', us_videos['category_id'].astype(str).map(categories))
    print(us_videos.info()) # check new column

    # percentage of dislikes
    us_videos['dislike_percentage'] = us_videos['dislikes'] / (us_videos['dislikes'] + us_videos['likes'])

    # Because many of the films have been trending several times, we drop duplicates
    us_videos_last = us_videos.drop_duplicates(subset=['video_id'], keep='last', inplace=False)
    us_videos_first = us_videos.drop_duplicates(subset=['video_id'], keep='first', inplace=False)

    # Check, original and subsets
    #print("us_videos dataset contains {} videos".format(us_videos.shape[0]))
    #print("us_videos_first dataset contains {} videos".format(us_videos_first.shape[0]))
    #print("us_videos_last dataset contains {} videos".format(us_videos_last.shape[0]))
    print(us_videos_first.head(1))
    print('shape: ', us_videos_first.shape)
    print('describe: ', us_videos_first.describe())
    # Create a time_to_trend; trending in days
    us_videos_first["time_to_trend"] = us_videos_first.trending_date - us_videos_first.publish_date

    publish = PlotTrend()
    # bar graph of best publishing hour
    # publish.publish_time(us_videos_first)

    # top 10 html display
    publish.publish_htm(us_videos)

    # correlation
    # publish.trend_corr(us_videos)

if __name__ == '__main__':
    main(sys.argv)
