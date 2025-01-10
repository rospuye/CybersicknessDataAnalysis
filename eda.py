# def visualize_heartbeat(filepath):
#     with open(filepath) as json_file:
#         data = json.load(json_file)

#     scenarios = list(data.keys())
#     print(data[scenarios[0]])

#     hrv_values = [data[scenario]['hrv'] for scenario in scenarios]
#     bpm_values = [data[scenario]['average_bpm'] for scenario in scenarios]

#     # print(hrv_values)
#     # print(bpm_values)

#     df = pd.DataFrame({
#         'Scenario': scenarios,
#         'HRV': hrv_values,
#         'Average_BPM': bpm_values
#     })

#     df['Avatar'] = df['Scenario'].apply(lambda x: 'Avatar' if 'avatar' in x else 'No Avatar')

#     # plt.figure(figsize=(10, 6))
#     # sns.barplot(x='Scenario', y='HRV', data=df, palette='muted')
#     # plt.title('Heart Rate Variability (HRV) Across Scenarios')
#     # plt.xticks(rotation=45, ha='right')
#     # plt.ylabel('HRV')
#     # plt.show()

#     # plt.figure(figsize=(8, 6))
#     # sns.boxplot(x='Avatar', y='HRV', data=df, palette='Set2')
#     # plt.title('Comparison of HRV Between Avatar and No Avatar Conditions')
#     # plt.ylabel('HRV')
#     # plt.show()

#     # plt.figure(figsize=(10, 6))
#     # sns.barplot(x='Scenario', y='Average_BPM', data=df, palette='coolwarm')
#     # plt.title('Average BPM Across Scenarios')
#     # plt.xticks(rotation=45, ha='right')
#     # plt.ylabel('Average BPM')
#     # plt.show()

#     # plt.figure(figsize=(8, 6))
#     # sns.boxplot(x='Avatar', y='Average_BPM', data=df, palette='coolwarm')
#     # plt.title('Comparison of Average BPM Between Avatar and No Avatar Conditions')
#     # plt.ylabel('Average BPM')
#     # plt.show()