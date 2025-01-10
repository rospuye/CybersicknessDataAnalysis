import glob
import os
import json
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel


def anova_ssq(filepath):
    with open(filepath) as json_file:
        ssq_data = json.load(json_file)

    data = []
    for participant_id, details in ssq_data.items():
        if (len(details.keys()) < 6):
            continue
        for condition in ['NS', 'AS', 'NW', 'AW']:
            row = {
                'participant': participant_id,
                'condition': condition,
                'score': sum(details[condition].values())
            }
            data.append(row)

    df = pd.DataFrame(data)
    aov = AnovaRM(data=df, depvar='score', subject='participant', within=['condition'])
    anova_results = aov.fit()
    posthoc = pairwise_tukeyhsd(df['score'], df['condition'], alpha=0.05)
    
    # print(anova_results)
    # print()
    # print(posthoc)

    #                 Anova
    # =======================================
    #           F Value Num DF  Den DF Pr > F
    # ---------------------------------------
    # condition  4.5975 3.0000 81.0000 0.0051
    # =======================================


    # Multiple Comparison of Means - Tukey HSD, FWER=0.05
    # ===================================================
    # group1 group2 meandiff p-adj   lower  upper  reject
    # ---------------------------------------------------
    #     AS     AW   0.7857 0.6897 -1.0766  2.648  False
    #     AS     NS   0.1071 0.9988 -1.7551 1.9694  False
    #     AS     NW   1.0357 0.4705 -0.8266  2.898  False
    #     AW     NS  -0.6786 0.7774 -2.5408 1.1837  False
    #     AW     NW     0.25 0.9852 -1.6123 2.1123  False
    #     NS     NW   0.9286 0.5642 -0.9337 2.7908  False
    # ---------------------------------------------------

    return anova_results, posthoc


def paired_ttests_heartbeat(folderpaths):

    data = []
    for path in folderpaths:
        with open(path) as json_file:
            user_data = json.load(json_file)
            data.append(user_data)

    # extract HRV and BPM for each condition
    hrv_seated_noavatar = [p["SCENARIO_seated_noavatar"]["hrv"] for p in data]
    hrv_seated_avatar = [p["SCENARIO_seated_avatar"]["hrv"] for p in data]
    hrv_walking_noavatar = [p["SCENARIO_walking_noavatar"]["hrv"] for p in data]
    hrv_walking_avatar = [p["SCENARIO_walking_avatar"]["hrv"] for p in data]

    bpm_seated_noavatar = [p["SCENARIO_seated_noavatar"]["average_bpm"] for p in data]
    bpm_seated_avatar = [p["SCENARIO_seated_avatar"]["average_bpm"] for p in data]
    bpm_walking_noavatar = [p["SCENARIO_walking_noavatar"]["average_bpm"] for p in data]
    bpm_walking_avatar = [p["SCENARIO_walking_avatar"]["average_bpm"] for p in data]

    # paired t-tests for HRV
    t_stat_hrv_seated, p_value_hrv_seated = ttest_rel(hrv_seated_noavatar, hrv_seated_avatar)
    t_stat_hrv_walking, p_value_hrv_walking = ttest_rel(hrv_walking_noavatar, hrv_walking_avatar)

    # paired t-tests for BPM
    t_stat_bpm_seated, p_value_bpm_seated = ttest_rel(bpm_seated_noavatar, bpm_seated_avatar)
    t_stat_bpm_walking, p_value_bpm_walking = ttest_rel(bpm_walking_noavatar, bpm_walking_avatar)

    # # display results
    # print("HRV Seated: t-stat =", t_stat_hrv_seated, ", p-value =", p_value_hrv_seated)
    # print("HRV Walking: t-stat =", t_stat_hrv_walking, ", p-value =", p_value_hrv_walking)
    # print("BPM Seated: t-stat =", t_stat_bpm_seated, ", p-value =", p_value_bpm_seated)
    # print("BPM Walking: t-stat =", t_stat_bpm_walking, ", p-value =", p_value_bpm_walking)

    return {
        "hrv_seated": {
            "t_stat": t_stat_hrv_seated,
            "p_value": p_value_hrv_seated
        },
        "hrv_walking": {
            "t_stat": t_stat_hrv_walking,
            "p_value": p_value_hrv_walking
        },
        "bpm_seated": {
            "t_stat": t_stat_bpm_seated,
            "p_value": p_value_bpm_seated
        },
        "bpm_walking": {
            "t_stat": t_stat_bpm_walking,
            "p_value": p_value_bpm_walking
        }
    }





if __name__ == "__main__":

    folder_path = "APP_DATA"
    xml_file = "export.xml"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    heartbeat_json_files = []

    for application_file in csv_files:
        id = os.path.splitext(os.path.basename(application_file))[0]
        metrics_folder_name = f"METRICS/{id}"

        heartbeat_metrics_filepath = f'{metrics_folder_name}/heartbeat_metrics.json'
        heartbeat_json_files.append(heartbeat_metrics_filepath)

    #     head_movement_metrics_filepath = f'{metrics_folder_name}/head_movement_metrics.json'

    anova_results, posthoc = anova_ssq('ssq.json') # for H1 and H3
    paired_ttests = paired_ttests_heartbeat(heartbeat_json_files) # for H2



