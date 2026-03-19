import pandas as pd

res_path_1 = "llama3b_2e_0.6t_auto_rewards.csv"
res_path_2 = "llama3b_2e_1.2t_auto_rewards.csv"
res_path_3 = "llama3b_2e_1.8t_auto_rewards.csv"
res_path_4 = "llama3b_2e_unsloth_res_rewards.csv"
new_data_path = "llama3b_2e_dpo_data_new_rewards_sbr.csv"

df1 = pd.read_csv(res_path_1)
df2 = pd.read_csv(res_path_2)
df3 = pd.read_csv(res_path_3)
df4 = pd.read_csv(res_path_4)

## Safety Rewards
df1["score"] =(100 * (df1["C"] + df1["T"]) - 1.5 * (df1["H"] + df1["M"]))
df2["score"] =(100 * (df2["C"] + df2["T"]) - 1.5 * (df2["H"] + df2["M"]))
df3["score"] =(100 * (df3["C"] + df3["T"]) - 1.5 * (df3["H"] + df3["M"]))
df4["score"] =(100 * (df4["C"] + df4["T"]) - 1.5 * (df4["H"] + df4["M"]))

df_new = df1[["input","output","instruction"]]

data = []
for i in range(df1.shape[0]):
    data.append([(df1.loc[i]["Generated"],df1.loc[i]["score"])])
    
for i in range(df2.shape[0]):
    data[i].append((df2.loc[i]["Generated"],df2.loc[i]["score"]))

for i in range(df3.shape[0]):
    data[i].append((df3.loc[i]["Generated"],df3.loc[i]["score"]))

for i in range(df3.shape[0]):
    data[i].append((df3.loc[i]["Generated"],df3.loc[i]["score"]))

for i in range(len(data)):
    data[i] = sorted(data[i],key = lambda x:x[1],reverse=True)

d1 = []
d2 = []
d3 = []
d4 = []
for i in data:
    d1.append(i[0][0])
    d2.append(i[1][0])
    d3.append(i[2][0])
    d4.append(i[3][0])

df_new["r1"] = d1
df_new["r2"] = d2
df_new["r3"] = d3
df_new["r4"] = d4

df_new.to_csv(new_data_path,index=False)