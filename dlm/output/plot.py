import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("cscw_results.csv")
df["approach"] = ["majority vote", "eq4", "eq3"]
df = df.melt("approach")
df["variable"] = df["variable"].apply(lambda x: x.replace("_", " ").title())
df['variable'] =[v if v!='Roc Auc' else 'ROC AUC' for v in df['variable']]

plt.figure()
ax = sns.barplot(x="value", y="variable", data=df, hue="approach")
for lab in ax.yaxis.get_ticklabels():
    lab.set_verticalalignment("center")
plt.legend(loc=(1.04,0))
plt.tight_layout()
plt.savefig("result-plot.pgf")
plt.show()


plt.figure()
ax = sns.barplot(x="value", y="variable", data=df[df['variable'].isin(
    ['Balanced Accuracy', 'Precision','Recall','F1', 'F1 Macro', 'ROC AUC', 'Jaccard']
    )], hue="approach")
for lab in ax.yaxis.get_ticklabels():
    lab.set_verticalalignment("center")
plt.legend(loc=(1.04,0))
plt.tight_layout()
plt.savefig("result-plot-subset.pgf")
plt.show()
