import os
import json
import ijson                                                    # type: ignore
import calendar
import numpy as np
import pandas as pd
import seaborn as sns                                           # type: ignore
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer     # type: ignore
from sklearn.decomposition import NMF                           # type: ignore
from wordcloud import WordCloud                                 # type: ignore

# Resolve the path to the static folder relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

class YouTubeReader:
    def __init__(self, file_path):
        """
        Initialize the class with the file path of the JSON file.
        """
        self.file_path = file_path
        self.raw_data = None
        self.dataframe = None

    def stream_JSON(self):
        """
        Stream large JSON files iteratively.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for item in ijson.items(file, 'item'):
                    yield item
        except Exception as e:
            raise FileNotFoundError(f"Error streaming JSON file: {e}")

    def load_JSON(self):
        """
        Load the JSON file and store the raw data
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.raw_data = json.load(file)
        except Exception as e:
            raise FileNotFoundError(f"Error loading JSON file: {e}")

    def to_dataframe(self):
        """
        Convert the raw JSON data into a pandas DataFrame using a general solution.
        """
        try:
            if not self.raw_data:
                raise ValueError("No data loaded. Run 'load_json()' first.")
            
            # Flatten the JSON into a DataFrame
            df = pd.json_normalize(self.raw_data)
            
            # Split 'subtitles' into 'channel_name' and 'channel_url'
            if 'subtitles' in df.columns:
                subtitles_df = df['subtitles'].apply(lambda x: x[0] if isinstance(x, list) and x else {}).apply(pd.Series)
                df['channel_name'] = subtitles_df.get('name')
                df['channel_url'] = subtitles_df.get('url')
                df.drop(columns=['subtitles'], inplace=True)
            
            # Split 'details' into 'details_name'
            if 'details' in df.columns:
                details_df = df['details'].apply(lambda x: x[0] if isinstance(x, list) and x else {}).apply(pd.Series)
                df['details_name'] = details_df.get('name')
                df.drop(columns=['details'], inplace=True)
            
            # Store the cleaned DataFrame
            self.dataframe = df
        except KeyError as e:
            print(f"KeyError in to_dataframe: {e}")
            self.dataframe = pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred in to_dataframe: {e}")
            self.dataframe = pd.DataFrame()

    def remove_Ads(self):
        """
        Remove observations that are Google Ads.
        """
        try:
            if self.dataframe is None:
                raise ValueError("Data has not been processed yet. Run 'to_dataframe()' first.")
            
            # Remove rows with 'From Google Ads' in 'details_name'
            self.dataframe = self.dataframe[self.dataframe['details_name'] != 'From Google Ads']
        except KeyError as e:
            print(f"KeyError in remove_Ads: {e}. 'details_name' column may be missing.")
        except ValueError as e:
            print(f"ValueError in remove_Ads: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in remove_Ads: {e}")

    def get_dataframe(self):
        """
        Return the processed DataFrame with only specific columns: 'title', 'time', and 'channel_name'.
        """
        try:
            if self.dataframe is None:
                raise ValueError("Data has not been processed yet. Run 'to_dataframe()' first.")
            
            #  Columns to keep
            columns_to_keep = ['title', 'time', 'channel_name']
            
            # Check for unnecessary columns and drop them
            current_columns = self.dataframe.columns
            columns_to_drop = [col for col in current_columns if col not in columns_to_keep]
            
            # Attempt to drop columns
            self.dataframe = self.dataframe.drop(columns=columns_to_drop, errors='ignore')
            
            # Ensure only the columns of interest remain
            self.dataframe = self.dataframe[columns_to_keep]
            return self.dataframe
        
        except KeyError as e:
            print(f"KeyError: {e}. Some columns to drop might not exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


class YouTubeWrangler:
    def __init__(self, data):
        """
        Initialize the class with a pandas DataFrame.
        """
        self.data = data.copy()
    
    # Decorators for exception handling
    def _handle_non_visual(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in non-visual function '{func.__name__}': {e}")
                return pd.DataFrame()  # Return an empty DataFrame
        return wrapper

    def _handle_table_analytics(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in table/text analytics function '{func.__name__}': {e}")
                return pd.DataFrame({"Error": ["An error occurred"]})  # Return a placeholder table
        return wrapper

    def _handle_plotting(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in plotting function '{func.__name__}': {e}")
        return wrapper

    @_handle_non_visual
    def discrete_tseries(self):
        """
        Process the 'time' column to extract year, month, day, and video counts.
        Returns a DataFrame with 'year', 'month', 'day', 'video_count', and 'year_and_month' columns.
        """
        self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce')
        df_copy = self.data.dropna(subset=['time']).copy()  # Drop rows where 'time' is NaT
        df_copy['year'] = df_copy['time'].dt.year
        df_copy['month'] = df_copy['time'].dt.month
        df_copy['day'] = df_copy['time'].dt.day  # Add days to the dataset

        # Group by year and month to count videos
        grouped = df_copy.groupby(['year', 'month']).size().reset_index(name='video_count')
        grouped['year_and_month'] = grouped.apply(
            lambda row: f"{calendar.month_abbr[int(row['month'])]} {int(row['year'])}", axis=1
        )
        return grouped
    
    @_handle_non_visual
    def continuous_tseries(self):
        """
        Add five new columns derived from the 'time' column:
        - 'year': Year of observation.
        - 'month': Month name from January to December.
        - 'day': Day of the week (Monday to Sunday).
        - 'time_of_day': Exact time the observation took place (HH:MM:SS).
        - 'hour': Hour range (0-23).
        Returns the updated DataFrame.
        """
        self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce')
        df_copy = self.data.copy() # Make a copy without dropping NaT values
        
        # Add the new columns, filling NaT-derived rows with placeholders
        df_copy['year'] = df_copy['time'].dt.year.fillna("Unknown")
        df_copy['month'] = df_copy['time'].dt.month_name().fillna("Unknown")
        df_copy['day'] = df_copy['time'].dt.day_name().fillna("Unknown")
        df_copy['time_of_day'] = df_copy['time'].dt.strftime('%H:%M:%S').fillna("Unknown")
        df_copy['hour'] = df_copy['time'].dt.hour.fillna(-1).astype(int)  # Use -1 for missing hours
        
        return df_copy
    
    @_handle_table_analytics
    def watch_stats(self):
        """
        Display the minimum and maximum values for yearly and monthly watched YouTube videos.
        Returns a pandas DataFrame with the stats.
        """
        # Extract time series data
        tseries = self.discrete_tseries()
        
        # Group by year and sum video counts
        yearly_counts = tseries.groupby('year')['video_count'].sum()
        min_year = yearly_counts.idxmin()
        max_year = yearly_counts.idxmax()
        
        # Find year and month with least and most videos watched
        min_month = tseries.loc[tseries['video_count'].idxmin()]
        max_month = tseries.loc[tseries['video_count'].idxmax()]

        # Create a summary table
        stats = {
            " ": [
                "Year with least videos watched",
                "Year with most videos watched",
                "Month with least videos watched",
                "Month with most videos watched"
            ],
            "Time Period": [
                min_year,
                max_year,
                f"{calendar.month_abbr[min_month['month']]} {min_month['year']}",
                f"{calendar.month_abbr[max_month['month']]} {max_month['year']}"
            ],
            "Video Count": [
                yearly_counts[min_year],
                yearly_counts[max_year],
                min_month['video_count'],
                max_month['video_count']
            ]
        }
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        return stats_df

    @_handle_table_analytics
    def top_seasons(self):
        """
        Sort all observations into seasons, aggregate video counts for each season,
        and display a table sorted by the number of videos watched with 1-based indexing.
        """
        df = self.discrete_tseries()
        
        # Map months to seasons
        seasons = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 
            4: 'Spring', 5: 'Spring', 6: 'Summer', 
            7: 'Summer', 8: 'Summer', 9: 'Autumn', 
            10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        
        df['season'] = df['month'].map(seasons)
        
        # Aggregate by seasons
        season_agg = df.groupby('season')['video_count'].sum().reset_index()
        season_agg.rename(columns={'video_count': 'Videos Watched Count'}, inplace=True)
        
        # Sort by Videos Watched Count in descending order
        season_agg = season_agg.sort_values(by='Videos Watched Count', ascending=False).reset_index(drop=True)
        
        # Change to 1-based indexing
        season_agg.index = season_agg.index + 1
        return season_agg

    @_handle_plotting
    def lineplt(self, save_path=None):
        """
        Generate a line plot for videos watched per month using Seaborn, 
        but display simplified x-axis labels as quarters.
        """
        df = self.discrete_tseries()
        
        # Create a 'date' column to sort and plot properly
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values(by='date', kind='mergesort') # Sorts 
        
        # Map each date to its corresponding quarter label
        df['quarter'] = df['date'].dt.to_period('Q').astype(str)
        
        # Prepare x-ticks: one tick per quarter (keep the rest hidden)
        quarters = df['quarter']
        unique_quarters = sorted(quarters.unique())
        quarter_ticks = [quarters[quarters == q].index[0] for q in unique_quarters]
        
        # Plot the line chart using Seaborn
        plt.figure(figsize=(18, 9))
        sns.lineplot(x=range(len(df['date'])), y=df['video_count'], marker='o', color='red', linewidth=2, label='Videos Watched')
        
        # Customize the x-axis with quarter labels
        plt.xticks(quarter_ticks, unique_quarters, rotation=45)
        plt.ylim(0)  # Ensure y-axis starts at 0
        plt.grid(True, linewidth=0.1)
        plt.xlabel("Quarter & Year")
        plt.ylabel("Number of Watched YouTube Videos")
        plt.title("YouTube Videos Watched Per Month (Labeled by Quarter)")
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @_handle_plotting
    def top_channels(self, save_path=None):
        """
        Generate a barplot of the top 10 channels based on the number of videos watched.
        """
        # Group and sort the data
        views_channel = self.data.groupby(self.data['channel_name'], dropna=False)['channel_name'].count()
        views_channel = views_channel.sort_values(ascending=False, kind='heapsort').head(10)  # Top 20 channels

        # Define the custom colormap using 5 colors
        custom_colors = ['#8B0000', '#C0392B', '#FF0000', '#FFB6C1', '#E8E8E8']
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

        # Normalize the video counts for the color range
        norm = mcolors.Normalize(vmin=views_channel.min(), vmax=views_channel.max())
        colors = [cmap(1 - norm(value)) for value in views_channel]

        # Plot the barplot
        plt.figure(figsize=(30, 16))
        ax = sns.barplot(
            x=views_channel.values, 
            y=views_channel.index, 
            hue=views_channel.index,  # Assign hue to y-axis (channel names)
            palette=colors,
            legend=False  # Suppress the unnecessary legend
        )

        # Set x-axis and y-axis labels
        ax.set_xlabel('No. of videos watched', fontsize=20)
        ax.set_ylabel('Channel Name', fontsize=20)

        # Set a title
        plt.title('Your Top 10 Most Watched YT-Channels', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    @_handle_plotting
    def heatmap(self, save_path=None):
        """
        Generate a heatmap showing the number of videos watched for each hour of the day and each day of the week.
        """
        df = self.continuous_tseries()
        
        # Filter out invalid hours (e.g., -1)
        df = df[df['hour'] >= 0]
        
        # Create a pivot table
        heatmap_data = df.pivot_table(index='day', columns='hour', aggfunc='size', fill_value=0)
        
        # Reorder days to start from Monday
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        # Plot the heatmap
        plt.figure(figsize=(15, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Number of Videos Watched'})
        plt.title("YouTube Watch History Heatmap")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Day of the Week")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()


class YouTubeTextStats:
    def __init__(self, data):
        """
        Initialize the class with a pandas DataFrame.
        """
        self.data = data.copy()

    # Decorators for exception handling title_analyzer and topic_modelling
    def _handle_text_analytics(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
        return wrapper

    def clean_titles(self):
        """
        Clean the DataFrame to only include the 'title' column and remove generic words.
        Returns a DataFrame with cleaned 'title' column.
        """
        try:
            cleaned_df = self.data[['title']].copy()
            cleaned_df['title'] = cleaned_df['title'].str.replace(r'^Watched ', '', regex=True)
            cleaned_df['title'] = cleaned_df['title'].str.replace(r'\b(shorts|watch|youtube|www|com|http|https|video|removed|videos removed)\b', '', regex=True)
            cleaned_df['title'] = cleaned_df['title'].str.replace(r'\bhttps?://\S+\b', '', regex=True)
            return cleaned_df
        except KeyError as e:
            print(f"KeyError in clean_titles: {e}. 'title' column may be missing.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred in clean_titles: {e}")
            return pd.DataFrame()

    @_handle_text_analytics
    def title_analyzer(self, top_n=10):
        """
        Analyze the 'title' column using TF-IDF and extract the top N keywords or phrases.
        Prints the most important keywords and their TF-IDF scores.
        """
        cleaned_df = self.clean_titles()
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(cleaned_df['title'])
        feature_names = vectorizer.get_feature_names_out()

        # Summarize most important terms
        tfidf_sum = tfidf_matrix.sum(axis=0)
        keywords = [(feature_names[i], tfidf_sum[0, i]) for i in range(len(feature_names))]
        sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
        
        print("Top Keywords and Phrases:")
        for word, score in sorted_keywords:
            print(f"{word}: {score:.2f}")

    @_handle_text_analytics
    def topic_modelling(self, num_topics=5, num_words=10):
        """
        Perform topic modelling on the 'title' column using Gensim's NMF model.
        Prints the identified topics with their top words.
        """
        cleaned_df = self.clean_titles()
        titles = cleaned_df['title'].dropna().tolist()
        
        # Vectorize the cleaned titles using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(titles)
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply Non-negative Matrix Factorization (NMF) for topic modeling
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_matrix = nmf_model.fit_transform(tfidf_matrix)
        
        # Display the topics with their top words
        print("Top Topics Identified:")
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    def cloud(self, save_path=None):
        """
        Generate a word cloud from the 'title' column.
        """
        try:
            cleaned_df = self.clean_titles()
            text = ' '.join(cleaned_df['title'].dropna())

            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                  background_color='#fffcfa', 
                                  colormap='Reds').generate(text)

            # Plot the word cloud
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of YouTube Titles', fontsize=18)

            # Set the save path explicitly to the static folder
            if save_path is None:
                save_path = os.path.join(STATIC_FOLDER, 'wordcloud.png')

            # Create the static folder if it does not exist
            os.makedirs(STATIC_FOLDER, exist_ok=True)

            # Save the plot
            plt.savefig(save_path)
            plt.close()
        except ValueError as e:
            print(f"ValueError in cloud: {e}. Ensure titles are cleaned and non-empty.")
        except Exception as e:
            print(f"An unexpected error occurred in cloud: {e}")
