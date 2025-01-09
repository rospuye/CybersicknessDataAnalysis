import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_heartbeat(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    print(data)

if __name__ == "__main__":

    folder_path = "APP_DATA"
    xml_file = "export.xml"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for application_file in csv_files:
        id = os.path.splitext(os.path.basename(application_file))[0]
        metrics_folder_name = f"METRICS/{id}"

        head_movement_metrics_filepath = f'{metrics_folder_name}/head_movement_metrics.json'
        heartbeat_metrics_filepath = f'{metrics_folder_name}/heartbeat_metrics.json'

        visualize_heartbeat(heartbeat_metrics_filepath)




