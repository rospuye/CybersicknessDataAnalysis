import glob
import os
import json
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def repeated_measures_anova(filepath):
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
    
    print(anova_results)
    print()
    print(posthoc)

    #                  Anova
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

    plot_anova_results(df)
    plot_boxplot_with_individual_points(df)
    return anova_results, posthoc


def paired_ttests(folderpaths):

    data = []
    for path in folderpaths:
        with open(path) as json_file:
            user_data = json.load(json_file)
            data.append(user_data)

    # extract HRV and BPM for each condition
    hrv_seated_noavatar = np.array([p["SCENARIO_seated_noavatar"]["hrv"] for p in data])
    hrv_seated_avatar = np.array([p["SCENARIO_seated_avatar"]["hrv"] for p in data])
    hrv_walking_noavatar = np.array([p["SCENARIO_walking_noavatar"]["hrv"] for p in data])
    hrv_walking_avatar = np.array([p["SCENARIO_walking_avatar"]["hrv"] for p in data])

    bpm_seated_noavatar = np.array([p["SCENARIO_seated_noavatar"]["average_bpm"] for p in data])
    bpm_seated_avatar = np.array([p["SCENARIO_seated_avatar"]["average_bpm"] for p in data])
    bpm_walking_noavatar = np.array([p["SCENARIO_walking_noavatar"]["average_bpm"] for p in data])
    bpm_walking_avatar = np.array([p["SCENARIO_walking_avatar"]["average_bpm"] for p in data])

    # replace NaN with mean value
    hrv_seated_noavatar = np.where(np.isnan(hrv_seated_noavatar), np.nanmean(hrv_seated_noavatar), hrv_seated_noavatar)
    hrv_seated_avatar = np.where(np.isnan(hrv_seated_avatar), np.nanmean(hrv_seated_avatar), hrv_seated_avatar)
    hrv_walking_noavatar = np.where(np.isnan(hrv_walking_noavatar), np.nanmean(hrv_walking_noavatar), hrv_walking_noavatar)
    hrv_walking_avatar = np.where(np.isnan(hrv_walking_avatar), np.nanmean(hrv_walking_avatar), hrv_walking_avatar)

    bpm_seated_noavatar = np.where(np.isnan(bpm_seated_noavatar), np.nanmean(bpm_seated_noavatar), bpm_seated_noavatar)
    bpm_seated_avatar = np.where(np.isnan(bpm_seated_avatar), np.nanmean(bpm_seated_avatar), bpm_seated_avatar)
    bpm_walking_noavatar = np.where(np.isnan(bpm_walking_noavatar), np.nanmean(bpm_walking_noavatar), bpm_walking_noavatar)
    bpm_walking_avatar = np.where(np.isnan(bpm_walking_avatar), np.nanmean(bpm_walking_avatar), bpm_walking_avatar)

    # paired t-tests for HRV
    t_stat_hrv_seated, p_value_hrv_seated = ttest_rel(hrv_seated_noavatar, hrv_seated_avatar)
    t_stat_hrv_walking, p_value_hrv_walking = ttest_rel(hrv_walking_noavatar, hrv_walking_avatar)

    # paired t-tests for BPM
    t_stat_bpm_seated, p_value_bpm_seated = ttest_rel(bpm_seated_noavatar, bpm_seated_avatar)
    t_stat_bpm_walking, p_value_bpm_walking = ttest_rel(bpm_walking_noavatar, bpm_walking_avatar)

    # display results
    print("HRV Seated: t-stat =", t_stat_hrv_seated, ", p-value =", p_value_hrv_seated)
    print("HRV Walking: t-stat =", t_stat_hrv_walking, ", p-value =", p_value_hrv_walking)
    print("BPM Seated: t-stat =", t_stat_bpm_seated, ", p-value =", p_value_bpm_seated)
    print("BPM Walking: t-stat =", t_stat_bpm_walking, ", p-value =", p_value_bpm_walking)
    print()


    # HRV Seated: t-stat = 1.4126374886815987 , p-value = 0.16962150200243586
    # HRV Walking: t-stat = -0.043670507440014765 , p-value = 0.9655007153146655
    # BPM Seated: t-stat = -0.9429630172488778 , p-value = 0.3543801247820677
    # BPM Walking: t-stat = -0.4701897604790042 , p-value = 0.6421396666174085
    
    graph_data = {
        "hrv_seated_noavatar": hrv_seated_noavatar,
        "hrv_seated_avatar": hrv_seated_avatar,
        "hrv_walking_noavatar": hrv_walking_noavatar,
        "hrv_walking_avatar": hrv_walking_avatar,
        "bpm_seated_noavatar": bpm_seated_noavatar,
        "bpm_seated_avatar": bpm_seated_avatar,
        "bpm_walking_noavatar": bpm_walking_noavatar,
        "bpm_walking_avatar": bpm_walking_avatar
    }

    plot_paired_scatter(graph_data, folderpaths)
    plot_difference_histogram(graph_data, folderpaths)


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

def two_way_repeated_measures_anova(filepath):
    with open(filepath) as json_file:
        ssq_data = json.load(json_file)

    data = []
    for participant_id, details in ssq_data.items():
        if len(details.keys()) < 6:
            continue
        for condition in ['NS', 'AS', 'NW', 'AW']:
            avatar = 'avatar' if 'A' in condition else 'no_avatar'
            locomotion = 'walking' if 'W' in condition else 'static'
            row = {
                'participant': participant_id,
                'avatar': avatar,
                'locomotion': locomotion,
                'condition': condition,
                'score': sum(details[condition].values())
            }
            data.append(row)

    df = pd.DataFrame(data)

    aov = AnovaRM(data=df, depvar='score', subject='participant', within=['avatar', 'locomotion'])
    anova_results = aov.fit()

    df['interaction'] = df['avatar'] + "_" + df['locomotion']
    posthoc = pairwise_tukeyhsd(df['score'], df['interaction'], alpha=0.05)

    print(anova_results)
    print()
    print(posthoc)

    #                      Anova
    # ===============================================
    #                   F Value Num DF  Den DF Pr > F
    # -----------------------------------------------
    # avatar             0.9414 1.0000 27.0000 0.3405
    # locomotion         6.6179 1.0000 27.0000 0.0159
    # avatar:locomotion  0.2222 1.0000 27.0000 0.6411
    # ===============================================


    #           Multiple Comparison of Means - Tukey HSD, FWER=0.05           
    # ========================================================================
    #      group1            group2      meandiff p-adj   lower  upper  reject
    # ------------------------------------------------------------------------
    #    avatar_static    avatar_walking   0.7857 0.6897 -1.0766  2.648  False
    #    avatar_static  no_avatar_static   0.1071 0.9988 -1.7551 1.9694  False
    #    avatar_static no_avatar_walking   1.0357 0.4705 -0.8266  2.898  False
    #   avatar_walking  no_avatar_static  -0.6786 0.7774 -2.5408 1.1837  False
    #   avatar_walking no_avatar_walking     0.25 0.9852 -1.6123 2.1123  False
    # no_avatar_static no_avatar_walking   0.9286 0.5642 -0.9337 2.7908  False
    # ------------------------------------------------------------------------

    plot_interaction(df)
    plot_factorial_boxplot(df)
    return anova_results, posthoc

# ---------------------------------------------------------------------------------

def sem(values):
    return np.std(values) / np.sqrt(len(values))

def plot_anova_results(df):
    mean_scores = df.groupby('condition')['score'].mean()
    sem_scores = df.groupby('condition')['score'].apply(sem)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='condition', y='score', data=df, ci="sd", palette="Set2", estimator=np.mean)
    plt.errorbar(x=range(len(mean_scores)), 
                 y=mean_scores, 
                 yerr=sem_scores, 
                 fmt='none', 
                 c='black', 
                 capsize=5, 
                 elinewidth=2)
    
    plt.title('Mean Scores Across Conditions with Error Bars')
    plt.xlabel('Scenario')
    plt.ylabel('Mean Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_boxplot_with_individual_points(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='condition', y='score', data=df, palette="Set2", fliersize=0)
    sns.stripplot(x='condition', y='score', data=df, color='black', alpha=0.5, jitter=True, size=5)

    plt.title('Distribution of Scores Across Conditions')
    plt.xlabel('Scenario')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------

def plot_paired_scatter(data, folderpaths):
    participant_ids = [os.path.basename(path).split('_')[0] for path in folderpaths]
    color_map = {participant: plt.cm.jet(i / len(participant_ids)) for i, participant in enumerate(participant_ids)}
    _, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].scatter(data["hrv_seated_noavatar"], data["hrv_seated_avatar"], c=[color_map[participant] for participant in participant_ids])
    axes[0, 0].plot([min(data["hrv_seated_noavatar"]), max(data["hrv_seated_noavatar"])], [min(data["hrv_seated_noavatar"]), max(data["hrv_seated_noavatar"])], 'k--', label="45-degree line")
    axes[0, 0].set_title("HRV: Seated (No Avatar) vs Seated (Avatar)")
    axes[0, 0].set_xlabel("Seated (No Avatar) HRV")
    axes[0, 0].set_ylabel("Seated (Avatar) HRV")
    axes[0, 0].legend(loc='upper left')

    axes[0, 1].scatter(data["bpm_seated_noavatar"], data["bpm_seated_avatar"], c=[color_map[participant] for participant in participant_ids])
    axes[0, 1].plot([min(data["bpm_seated_noavatar"]), max(data["bpm_seated_noavatar"])], [min(data["bpm_seated_noavatar"]), max(data["bpm_seated_noavatar"])], 'k--', label="45-degree line")
    axes[0, 1].set_title("BPM: Seated (No Avatar) vs Seated (Avatar)")
    axes[0, 1].set_xlabel("Seated (No Avatar) BPM")
    axes[0, 1].set_ylabel("Seated (Avatar) BPM")
    axes[0, 1].legend(loc='upper left')

    axes[1, 0].scatter(data["hrv_walking_noavatar"], data["hrv_walking_avatar"], c=[color_map[participant] for participant in participant_ids])
    axes[1, 0].plot([min(data["hrv_walking_noavatar"]), max(data["hrv_walking_noavatar"])], [min(data["hrv_walking_noavatar"]), max(data["hrv_walking_noavatar"])], 'k--', label="45-degree line")
    axes[1, 0].set_title("HRV: Walking (No Avatar) vs Walking (Avatar)")
    axes[1, 0].set_xlabel("Walking (No Avatar) HRV")
    axes[1, 0].set_ylabel("Walking (Avatar) HRV")
    axes[1, 0].legend(loc='upper left')

    axes[1, 1].scatter(data["bpm_walking_noavatar"], data["bpm_walking_avatar"], c=[color_map[participant] for participant in participant_ids])
    axes[1, 1].plot([min(data["bpm_walking_noavatar"]), max(data["bpm_walking_noavatar"])], [min(data["bpm_walking_noavatar"]), max(data["bpm_walking_noavatar"])], 'k--', label="45-degree line")
    axes[1, 1].set_title("BPM: Walking (No Avatar) vs Walking (Avatar)")
    axes[1, 1].set_xlabel("Walking (No Avatar) BPM")
    axes[1, 1].set_ylabel("Walking (Avatar) BPM")
    axes[1, 1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_difference_histogram(data, folderpaths):
    hrv_diff_seated = data["hrv_seated_avatar"] - data["hrv_seated_noavatar"]
    bpm_diff_seated = data["bpm_seated_avatar"] - data["bpm_seated_noavatar"]
    hrv_diff_walking = data["hrv_walking_avatar"] - data["hrv_walking_noavatar"]
    bpm_diff_walking = data["bpm_walking_avatar"] - data["bpm_walking_noavatar"]
    _, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].hist(hrv_diff_seated, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Difference')
    axes[0, 0].set_title("HRV Difference: Seated (Avatar) - Seated (No Avatar)")
    axes[0, 0].set_xlabel("Difference in HRV")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    axes[0, 1].hist(bpm_diff_seated, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Difference')
    axes[0, 1].set_title("BPM Difference: Seated (Avatar) - Seated (No Avatar)")
    axes[0, 1].set_xlabel("Difference in BPM")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    axes[1, 0].hist(hrv_diff_walking, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Difference')
    axes[1, 0].set_title("HRV Difference: Walking (Avatar) - Walking (No Avatar)")
    axes[1, 0].set_xlabel("Difference in HRV")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    axes[1, 1].hist(bpm_diff_walking, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Difference')
    axes[1, 1].set_title("BPM Difference: Walking (Avatar) - Walking (No Avatar)")
    axes[1, 1].set_xlabel("Difference in BPM")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------

def plot_interaction(df):
    summary = df.groupby(['locomotion', 'avatar'])['score'].agg(['mean', 'sem']).reset_index()
    plt.figure(figsize=(8, 6))

    sns.pointplot(data=summary, x='locomotion', y='mean', hue='avatar', 
                  dodge=True, markers=["o", "s"], linestyles=["-", "--"],
                  capsize=0.1, ci=None, palette="Set1", join=True)

    plt.title('Interaction Plot: Avatar vs. Locomotion')
    plt.xlabel('Locomotion (Static, Walking)')
    plt.ylabel('Mean Score')
    plt.legend(title='Avatar', loc='upper left')
    plt.show()

def plot_factorial_boxplot(df):
    df['condition'] = df['avatar'] + "_" + df['locomotion']
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='condition', y='score', hue='avatar', palette="Set1")

    plt.title('Factorial Boxplot: Score Distribution across Avatar and Locomotion')
    plt.xlabel('Conditions (avatar_static, avatar_walking, no_avatar_static, no_avatar_walking)')
    plt.ylabel('Scores')
    plt.show()

# ---------------------------------------------------------------------------------

if __name__ == "__main__":

    folder_path = "APP_DATA"
    xml_file = "export.xml"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    heartbeat_json_files = []

    for application_file in csv_files:
        id = os.path.splitext(os.path.basename(application_file))[0]
        if (id in ["t12", "t16", "t19", "t26", "t27"]):
            continue

        metrics_folder_name = f"METRICS/{id}"

        heartbeat_metrics_filepath = f'{metrics_folder_name}/heartbeat_metrics.json'
        heartbeat_json_files.append(heartbeat_metrics_filepath)

    anova_results, posthoc = repeated_measures_anova('ssq.json') # for H1 and H3
    paired_ttests = paired_ttests(heartbeat_json_files) # for H2
    anova_results, posthoc = two_way_repeated_measures_anova('ssq.json') # for H4





