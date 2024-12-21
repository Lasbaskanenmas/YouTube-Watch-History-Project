# YouTube Watch History project

## Introduction
The YouTube Watch History project is a simple web-based tool that provides a general perspective of your YouTube viewing habits. It focuses on descriptive statistics such as viewing patterns over time and the most common topics seen in your YouTube history. The tool processes your YouTube data and presents the results through a Flask-based dashboard, which includes visualizations like:
- **Line Plot**: Monthly trends in video views.
- **Heatmap**: Hourly and weekly viewing patterns.
- **Bar Plot**: Top 10 most-watched channels.
- **Word Cloud**: Most frequent words in video titles.

The goal is not deep analytics but to give users an easy-to-understand overview of their YouTube habits.

---

## How to Get YouTube Data
1. Go to [Google Takeout](https://takeout.google.com).
2. Deselect all options and select **YouTube and YouTube Music**.
3. Click "All YouTube data included" and check only **History**.
4. Export the data in **JSON** format.
5. Download the ZIP file and extract it to find the `watch-history.json` file.

---

## Libraries Required
Before running the program, ensure the following Python libraries are installed:

- `json`
- `ijson`
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `wordcloud`
- `threading`

### Installation Command
Run the following command in your terminal:
```bash
pip install json ijson numpy pandas seaborn matplotlib scikit-learn wordcloud
```

If the above does not work, try the following snippet:
```python
import os
os.system('pip install json ijson numpy pandas seaborn matplotlib scikit-learn wordcloud')
```
*Note: This snippet attempts installation but may not work in all environments.*

---

## How to Run the Application
1. Place the `watch-history.json` file in the project directory.
2. Open and run the `main.ipynb` Jupyter notebook.
3. The notebook will:
   - Clean and process the JSON data.
   - Generate visualizations for time-based trends, top channels, and common topics.
   - Display general descriptive statistics, including:
     - **Seasonal Trends**: Shows the total videos watched in each season (e.g., Winter, Summer, Autumn, Spring).
     - **Top Topics**: Identifies the most common themes in the video titles, such as recurring words or phrases.
   - Launch a **Flask-based dashboard**.

### Expected Output
Once the notebook runs successfully:
1. The dashboard is hosted locally and displays:
   - A **Line Plot** showing the monthly YouTube activity.
   - A **Heatmap** depicting the days and hours of highest activity.
   - A **Bar Plot** listing your top 10 channels.
   - A **Word Cloud** illustrating frequently occurring words from video titles.
2. **General Descriptive Statistics** are displayed directly in the notebook, summarizing seasonal video counts and the most common topics.
3. All generated images are saved to the `static/` folder.

---

## Key Notes
- The tool focuses on descriptive statistics to highlight general patterns in your YouTube habits.
- Ensure Python 3.9.12 or newer is installed.
- The JSON file structure is consistent when downloaded from Google Takeout.

Enjoy exploring your YouTube watch history!
