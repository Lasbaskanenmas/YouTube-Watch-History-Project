import os
import sys
import subprocess
from flask import Flask, render_template, send_file, redirect, url_for  # type: ignore
from threading import Thread
from library import YouTubeReader, YouTubeWrangler, YouTubeTextStats

# Resolve the path to the static folder relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

# Function to install missing libraries
def install_libraries(libraries):
    for lib in libraries:
        try:
            __import__(lib)  # Try importing the library
        except ImportError:
            print(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except ModuleNotFoundError:
            print(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        else:
            print(f"{lib} is already installed.")

class YouTubePlots:
    def __init__(self, file_path):
        """
        Initialize the class with the path to the JSON file.
        """
        self.file_path = file_path
        self.cleaned_data = None
        self.yt_wrangler = None
        self.text_stats = None

    def clean_data(self):
        """
        Step 1: Use YouTubeReader to clean and load the data.
        """
        try:
            reader = YouTubeReader(self.file_path)
            reader.load_JSON()       # Load JSON data
            reader.to_dataframe()    # Convert to DataFrame
            reader.remove_Ads()      # Remove ads
            self.cleaned_data = reader.get_dataframe()
        except FileNotFoundError as e:
            print(f"FileNotFoundError in clean_data: {e}. Ensure the JSON file exists.")
        except ValueError as e:
            print(f"ValueError in clean_data: {e}. Data might be improperly formatted.")
        except Exception as e:
            print(f"An unexpected error occurred in clean_data: {e}")

    def get_cleaned_data(self):
        """
        Return the cleaned data as a pandas DataFrame.
        """
        if self.cleaned_data is not None:
            return self.cleaned_data
        else:
            raise ValueError("No cleaned data available. Ensure 'run_pipeline()' has been executed.")

    def generate_visuals(self, save=False):
        """
        Step 2: Generate all visuals using YouTubeWrangler and YouTubeTextStats.
        """
        # Debug: Verify static folder path
        print(f"Saving plots to: {STATIC_FOLDER}")  # Added for debugging the static path (CHANGE)
        os.makedirs(STATIC_FOLDER, exist_ok=True)  # Ensure the static folder exists

        # Initialize YouTubeWrangler
        self.yt_wrangler = YouTubeWrangler(self.cleaned_data)
        try:
            if save:
                # Save plots explicitly to static folder
                self.yt_wrangler.lineplt(save_path=os.path.join(STATIC_FOLDER, "lineplot.png"))
                self.yt_wrangler.heatmap(save_path=os.path.join(STATIC_FOLDER, "heatmap.png"))
                self.yt_wrangler.top_channels(save_path=os.path.join(STATIC_FOLDER, "top_channels.png"))
            else:
                self.yt_wrangler.lineplt()
                self.yt_wrangler.heatmap()
                self.yt_wrangler.top_channels()
        except Exception as e:
            print(f"Error saving visuals: {e}")  # Added error handling for visuals saving (CHANGE)

        # Initialize YouTubeTextStats
        self.text_stats = YouTubeTextStats(self.cleaned_data)
        try:
            if save:
                self.text_stats.cloud(save_path=os.path.join(STATIC_FOLDER, "wordcloud.png"))
            else:
                self.text_stats.cloud()
        except Exception as e:
            print(f"Error saving word cloud: {e}")  # Added error handling for word cloud saving (CHANGE)

    def run_pipeline(self, save_visuals=False):
        """
        Run the entire pipeline: cleaning data and generating visuals.
        """
        try:
            self.clean_data()
            self.generate_visuals(save=save_visuals)
        except Exception as e:
            print(f"An error occurred during pipeline execution: {e}")

    def time_plots(self):
        """
        Step 2.1: Use YouTubeWrangler to create lineplot and heatmap in a single grid.
        """
        try:
            self.yt_wrangler = YouTubeWrangler(self.cleaned_data)
            self.yt_wrangler.lineplt()
            self.yt_wrangler.heatmap()
        except AttributeError as e:
            print(f"AttributeError in time_plots: {e}. Ensure 'clean_data' is run first.")
        except Exception as e:
            print(f"An unexpected error occurred in time_plots: {e}")


class YouTubeApp:
    def __init__(self):
        self.app = Flask(__name__)
        
        # Resolve the static folder relative to app.py
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.static_folder = os.path.join(BASE_DIR, 'static')
        os.makedirs(self.static_folder, exist_ok=True)  # Ensure the folder exists

        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/generate')
        def generate_visuals():
            # Use the absolute path to the JSON file
            pipeline = YouTubePlots(os.path.join(BASE_DIR, 'watch-history.json'))  # Fixed JSON file path (CHANGE)
            pipeline.run_pipeline(save_visuals=True)
            return redirect(url_for('dashboard'))

        @self.app.route('/image/<name>')
        def serve_image(name):
            # Serve the image file if it exists
            image_path = os.path.join(self.static_folder, name)
            if os.path.exists(image_path):
                return send_file(image_path, mimetype='image/png')
            else:
                return "Image not found", 404

    def run(self, host='127.0.0.1', port=5000):
        self.app.run(host=host, port=port, debug=False, threaded=True)











