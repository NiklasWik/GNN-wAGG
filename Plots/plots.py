import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
import pandas as pd 
from matplotlib import pyplot as plt
import os
here = os.path.dirname(os.path.abspath(__file__))
res = pd.read_csv(os.path.join(here, "Resultat.csv"), sep = ',', decimal=',')
res = res.replace(to_replace="pnorm", value="p-norm")
res = res.replace(to_replace="pnorm", value="p-norm")
lab = list(res.head())
pattern = res[res.DATA=="PATTERN"]
cluster = res[res.DATA=="CLUSTER"]
colors = ["#3498DB","#e74c3c","#34495e","#2ecc71"]
sns.set_palette(colors)
yvec = ["TEST_ACC", "TRAIN_ACC"]

for _,yuse in enumerate(yvec):
    g = sns.catplot(x="GNN", y=yuse, hue="AGG", kind="bar", data=pattern, ci="sd", capsize=0.1)
    g.set(ylim=(min(pattern[yuse])-(max(res[yuse])-min(pattern[yuse]))*.05, max(pattern[yuse])+(max(res[yuse])-min(pattern[yuse]))*.1))
    g.set_xlabels("Network")
    g.set_ylabels("Accuracy (%)")
    plt.title(yuse[:-4].title()+"ing accuracy" + " (PATTERN)")
plt.show()
