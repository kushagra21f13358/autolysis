# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Initialize OpenAI
openai.api_key = os.environ.get("AIPROXY_TOKEN")

def load_data(file_path):
    """Load the dataset and return a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(data):
    """Perform basic data analysis."""
    summary = {
        "head": data.head().to_dict(),
        "info": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_stats": data.describe().to_dict()
    }
    return summary

def visualize_data(data, output_dir):
    """Generate visualizations and save as PNG files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    charts = []
    
    # Correlation heatmap
    if data.select_dtypes(include="number").shape[1] > 1:
        corr = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        heatmap_path = f"{output_dir}/correlation_heatmap.png"
        plt.savefig(heatmap_path)
        charts.append(heatmap_path)
        plt.close()

    # Distribution of numerical columns
    for col in data.select_dtypes(include="number").columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True, bins=30)
        dist_path = f"{output_dir}/{col}_distribution.png"
        plt.savefig(dist_path)
        charts.append(dist_path)
        plt.close()

    return charts

def query_llm(prompt):
    """Send a prompt to the LLM and return the response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying LLM: {e}")
        sys.exit(1)

def narrate_story(data_summary, charts, output_file):
    """Generate a narrative using the LLM and write to README.md."""
    prompt = (
        "Write a Markdown document narrating the story of the analysis. "
        f"The data summary is: {data_summary}. "
        f"The charts generated are: {', '.join(charts)}. "
        "Include descriptions of the data, insights, and implications."
    )
    story = query_llm(prompt)

    with open(output_file, "w") as f:
        f.write(story)

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = file_path.split('.')[0]
    output_file = f"{output_dir}/README.md"

    data = load_data(file_path)
    summary = analyze_data(data)
    charts = visualize_data(data, output_dir)
    narrate_story(summary, charts, output_file)

    print(f"Analysis complete. Outputs saved in {output_dir}/")

if __name__ == "__main__":
    main()
