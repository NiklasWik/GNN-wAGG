import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
import pandas as pd 
from matplotlib import pyplot as plt
sheet_id = "1T66TbM4FE8WcSFyaXqV5FlY90DCyWp81"
sheet_name = "EasyExport"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
res = pd.read_csv(url, sep = ',', decimal=',')
res = res.replace(to_replace="pnorm", value="p-norm")
res = res.replace(to_replace="planar_sig", value="planar$_{\mathrm{sig}}$")
res = res.replace(to_replace="planar_tanh", value="planar$_{\mathrm{tanh}}$")
datasets = [res[res.DATA=="PATTERN"], res[res.DATA=="CLUSTER"]]
colors = ["#3498DB","#e74c3c","#FFD548","#2ecc71"]
yvec = ["TEST_ACC", "TRAIN_ACC"]

for _,yuse in enumerate(yvec):
    for _,dataset in enumerate(datasets):
        g = sns.catplot(x="GNN", y=yuse, hue="AGG", kind="bar", data=dataset, ci="sd", capsize=0.1, palette=colors)
        g.set(ylim=(min(dataset[yuse])-(max(res[yuse])-min(dataset[yuse]))*.05, max(dataset[yuse])+(max(res[yuse])-min(dataset[yuse]))*.1))
        g.set_xlabels("Network")
        g.set_ylabels("Accuracy (%)")
        plt.title(yuse[:-4].title()+"ing accuracy, (" + dataset.iloc[0]['DATA'] +")")
plt.show()
