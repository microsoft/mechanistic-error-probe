import os
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset-name", type=str, default="basketball_players")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory where the attention flow is saved.")
    return parser.parse_args()

args = config()

def extract_predictors(records):
    token_contribs = records.token_contrib_norms_constraints
    attention_weights = records.attention_weights_constraints
    indices = np.arange(len(attention_weights))
    corr = []
    predictors = defaultdict(list)
    
    for token_head_norms, idx in zip(token_contribs, indices):
        for constraint_idx in range(len(records["constraint"][idx])):
            constraint_contributions = token_head_norms[constraint_idx]
            constraint_weights = attention_weights[idx][constraint_idx]
            corr.append(int(records["gt_logprob"][idx]==records["pred_logprob"][idx]))
            predictors[r"$||a_{C,T}^{\ell, [h]}||$"].append(constraint_contributions.max(axis=2).reshape((1, -1)))
            predictors[r"$||A_{C,T}^{\ell, [h]}||$"].append(constraint_weights.max(axis=2).reshape((1, -1)))

    predictors = {k: np.array(v)[:, 0] for k,v in predictors.items()}
    predictors[r"$\hat{P}(\hat{Y}|X)$"] = np.array(records["pred_logprob"])
    y = np.array(corr)
    predictors["Majority"] = np.ones(y.shape[0]) if np.mean(y) > 0.5 else np.zeros(y.shape[0])
    predictors["Popularity"] = np.array(records.popularity)
    predictors["Combined"] = np.concatenate([predictors[r"$||A_{C,T}^{\ell, [h]}||$"], predictors[r"$\hat{P}(\hat{Y}|X)$"].reshape(-1, 1)], axis=1)
    return predictors, y
    
def get_metrics(y, score):
    roc_auc = roc_auc_score(y, score)
    bottom20_idx = np.argsort(score)[:int(score.shape[0]*0.2)]
    top20_idx =  np.argsort(-score)[:int(score.shape[0]*0.2)]
    risk_at_top20 = 1-y[top20_idx].mean()
    risk_at_bottom20 = 1-y[bottom20_idx].mean()

    return {r"AUROC$\textcolor{Green}{\mathbf{(\Uparrow)}}$": roc_auc, 
            r"$\text{Risk}_{\textrm{Top 20\%}}(\textcolor{Blue}{\mathbf{\Downarrow}})$": risk_at_top20, 
            r"$\text{Risk}_{\textrm{Bottom 20\%}}(\textcolor{Green}{\mathbf{\Uparrow}})$":risk_at_bottom20}


output_file = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_{args.dataset_name}.pkl")
records = pickle.load(open(output_file, "rb"))
predictors, y = extract_predictors(records)
performance_records = []
seed = 0
train_idx, test_idx = train_test_split(np.arange(y.shape[0]), test_size=0.5, stratify=y, random_state=seed)

for predictor, X in predictors.items():
    X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    
    if predictor == '$\\hat{P}(\\hat{Y}|X)$':
        performance = get_metrics(y_test, X_test)
    elif predictor in ['Majority', 'Popularity']:
        performance = get_metrics(y_test, X_test)
    else:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        
        performance = get_metrics(y_test, LogisticRegression(penalty="l1", solver="liblinear", C=0.05, max_iter=1000).fit(X_train, y_train).predict_proba(X_test)[:, 1])
    for metric, val in performance.items():
        performance_records.append({"Data": args.dataset_name,
                       "BaseRate": y_test.mean(), 
                       "Predictor": predictor, 
                        "seed": seed,
                        "Metric": metric,
                        "Value": val
                                   })

# Rendering figures
df_results = pd.DataFrame(performance_records)
metric_to_plot = r"AUROC$\textcolor{Green}{\mathbf{(\Uparrow)}}$"
metrics_fig = [metric_to_plot]
df_results = df_results[df_results.Metric.isin(metrics_fig)]
predictors_fig = ['$||A_{C,T}^{\ell, [h]}||$', 'Majority', r"$\hat{P}(\hat{Y}|X)$", "Popularity", "Combined"]
df_results = df_results[df_results.Predictor.isin(predictors_fig)]

name_map = {
    r"$\hat{P}(\hat{Y}|X)$": r"$\textsc{Confidence}$",
    '$||A_{C,T}^{\ell, [h]}||$': r"$\textsc{SAT-Probe}$",
    "Majority": r"$\textsc{Constant}$",
    "Popularity": r"$\textsc{Popularity}$",
    "Combined": r"$\textsc{Combined}$"
}
df_results['Predictor'] = df_results['Predictor'].apply(lambda x: name_map[x])
name_map_rev= {v: k for k,v in name_map.items()} 
color_map = {
    r"$\hat{P}(\hat{Y}|X)$": 'firebrick',  # Lighter green
    'Majority': 'darkorange',  # Lighter orange
    '$||A_{C,T}^{\ell, [h]}||$': 'royalblue',  # Lighter blue,
    "Popularity": '#BEBEBE',
    "Combined": "turquoise"
}
name_to_color = {k: color_map[name_map_rev[k]] for k, v in name_map_rev.items()}

plt.rcParams['text.usetex'] = True

custom_order = [r"$\textsc{SAT-Probe}$", r"$\textsc{Confidence}$", r"$\textsc{Popularity}$", r"$\textsc{Constant}$", r"$\textsc{Combined}$"]

fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

sns.barplot(data=df_results, y="Data", x="Value", hue="Predictor", ax=ax, palette=name_to_color, hue_order=custom_order)
ax.legend(loc="lower right", fontsize="medium", framealpha=0.9)
ax.set_xlabel("AUROC")
ax.set_title(args.model_name)
fig.tight_layout()
figure_path = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_{args.dataset_name}_performance.png")
fig.savefig(figure_path)