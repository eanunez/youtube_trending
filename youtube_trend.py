import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib import cm as cm

'''
Sample of Data Analytics on Youtube Trending Using Python
Copyright: Skalskip and Donyoe of Kaggle.com'''


class Plot_trend:

    def trend_corr(self, data):

        column_names = ['category_id','views', 'likes', 'dislikes','comment_count']
        corr = data.corr()
        selected_corr = corr.loc[column_names, column_names]
        print(selected_corr)

        fig = plt.figure()
        fig.suptitle('US-Video Correlation Matrix', y=0.97)
        ax1 = fig.add_subplot(111)
        #ax1.grid(True)
        cmap = cm.get_cmap('jet', 30)

        cax = ax1.imshow(selected_corr, interpolation="nearest", cmap=cmap )
        #plt.title('US-Video Correlation Matrix')
        labels = ['','category_id','views', 'likes', 'dislikes','comment_count']

        ax1.set_xticklabels([])
        ax1.set_yticklabels(labels, fontsize=6)

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(labels, fontsize=6)

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        fig.colorbar(cax, ticks=[.1, .5, .8, .90, .95, 1])

        plt.show()

    def publish_time(self, us_videos_first):
        # Initialization of the list storing counters for subsequent publication hours
        publish_h = [0] * 24

        for index, row in us_videos_first.iterrows():   # row iteration in pandas
            publish_h[row["publish_hour"]] += 1     # row item in 'publish hour' is matched to publish_h index, increments

        values = publish_h
        ind = np.arange(len(values))

        # Creating new plot
        fig = plt.figure(figsize=(20, 10))
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
    print(us_videos.head(1))    # first line
    print(us_videos.info())     # info of the head

    # Transforming Trending date column to datetime format
    us_videos['trending_date'] = pd.to_datetime(us_videos['trending_date'], format='%y.%d.%m').dt.date

    # Transforming Trending date column to datetime format and splitting into two separate ones
    publish_time = pd.to_datetime(us_videos['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    us_videos['publish_date'] = publish_time.dt.date
    us_videos['publish_time'] = publish_time.dt.time
    us_videos['publish_hour'] = publish_time.dt.hour

    # Check again for changes
    print(us_videos.head(1))

    # We'll use a very nice python featur - dictionary comprehension, to extract most important data from US_category_id.json
    categories = {category['id']: category['snippet']['title'] for category in us_videos_categories['items']}

    # Now we will create new column that will represent name of category
    us_videos.insert(4, 'category', us_videos['category_id'].astype(str).map(categories))
    print(us_videos['category'].head(5)) # check new column

    # percentage of dislikes
    us_videos['dislike_percentage'] = us_videos['dislikes'] / (us_videos['dislikes'] + us_videos['likes'])

    # Because many of the films have been trending several times, we drop duplicates
    us_videos_last = us_videos.drop_duplicates(subset=['video_id'], keep='last', inplace=False)
    us_videos_first = us_videos.drop_duplicates(subset=['video_id'], keep='first', inplace=False)

    # Check, original and subsets
    print("us_videos dataset contains {} videos".format(us_videos.shape[0]))
    print("us_videos_first dataset contains {} videos".format(us_videos_first.shape[0]))
    print("us_videos_last dataset contains {} videos".format(us_videos_last.shape[0]))

    # Create a time_to_trend; trending in days
    us_videos_first["time_to_trend"] = us_videos_first.trending_date - us_videos_first.publish_date

    publish = Plot_trend()
    # bar graph of best publishing hour
    # publish.publish_time(us_videos_first)

    # top 10 html display
    #publish.publish_html(us_videos)

    # correlation
    publish.trend_corr(us_videos)

if __name__ == '__main__':
    main(sys.argv)


