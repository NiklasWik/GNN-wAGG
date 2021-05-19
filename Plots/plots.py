import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
sns.set_theme(style="ticks", color_codes=True)
import pandas as pd 
import numpy as np
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

#Result plots
if True:
    yvec = ["TEST_ACC", "TRAIN_ACC"]
    for _,yuse in enumerate(yvec):
        for _,dataset in enumerate(datasets):
            g = sns.catplot(x="GNN", y=yuse, hue="AGG", kind="bar", data=dataset, ci="sd", capsize=0.1, palette=colors)
            g.set(ylim=(min(dataset[yuse])-(max(res[yuse])-min(dataset[yuse]))*.05, max(dataset[yuse])+(max(res[yuse])-min(dataset[yuse]))*.1))
            g.set_xlabels("Network")
            g.set_ylabels("Accuracy (%)")
            plt.title(yuse[:-4].title()+"ing accuracy, (" + dataset.iloc[0]['DATA'] +")")
            plt.savefig(str(pathlib.Path(__file__).parent.absolute())+"//"+yuse[:-4].replace(" ", "")+"-" + dataset.iloc[0]['DATA']+".eps", bbox_inches="tight", format="eps")

#Time plots
if True:
    yvec = ["EPOCHS", "EPOCH TIME (s)", "TOTAL TIME (h)"]
    ylabs = ["Number of epochs (#)", "Time per epoch (s)", "Total training time (h)"]
    for i,yuse in enumerate(yvec):
        for _,dataset in enumerate(datasets):
            g = sns.catplot(x="GNN", y=yuse, hue="AGG", kind="bar", data=dataset, ci="sd", capsize=0.1, palette=colors)
            g.set(ylim=(min(dataset[yuse])-(max(res[yuse])-min(dataset[yuse]))*.05, max(dataset[yuse])+(max(res[yuse])-min(dataset[yuse]))*.1))
            g.set_xlabels("Network")
            g.set_ylabels(ylabs[i])
            plt.title(yuse[0:10].title()+", (" + dataset.iloc[0]['DATA'] +")")
            plt.savefig(str(pathlib.Path(__file__).parent.absolute())+"//"+yuse[0:10].replace(" ", "")+"-" + dataset.iloc[0]['DATA']+".eps", bbox_inches="tight", format="eps")

#Tables
if True:
    gnns = ["GATED", "SAGE", "GAT", "GIN"]
    dset = ["PATTERN", "CLUSTER"]
    for j,data in enumerate(datasets):
        pd.options.display.float_format = '{:,.3f}'.format
        for i,df in enumerate([data[data.GNN==x] for x in gnns]):
            df = df.groupby(['AGG'], as_index=False).agg({'Layers':'mean', 'Params':'mean', 'TEST_ACC':['mean','std'], 'TRAIN_ACC':['mean','std'], 'EPOCHS':'mean', 'EPOCH TIME (s)':'mean', 'TOTAL TIME (h)':'mean'})
            for asd in ["TEST_ACC", "TRAIN_ACC"]:
                df[asd] = df[asd]['mean'].round(3).astype(str)+'±'+df[asd]['std'].round(3).astype(str)
            df = df.drop('std', axis=1, level=1)
            df.columns = df.columns.droplevel(1)
            df.Params = df.Params.round(0).astype(int)
            df['EPOCHS'] = df['EPOCHS'].round(2).astype(str)
            df['EPOCH TIME (s)'] = df['EPOCH TIME (s)'].round(2).astype(str)+'s/'+df['TOTAL TIME (h)'].round(2).astype(str)+'h'
            df = df.drop('TOTAL TIME (h)', axis = 1)
            df.AGG = df.AGG.astype(str)
            df.columns = ["\textbf{Agg. func.}", "\emph{L}", "\textbf{#Params}", "\textbf{Test Acc. ± s.d.}", "\textbf{Train Acc. ± s.d.}", "\textbf{#Epochs}", "\textbf{Epoch/Total}"] 
            df1 = df[["\textbf{Agg. func.}", "\emph{L}"]]
            df2 = df[["\textbf{#Params}", "\textbf{Test Acc. ± s.d.}", "\textbf{Train Acc. ± s.d.}", "\textbf{#Epochs}", "\textbf{Epoch/Total}"]]
            
            with open(str(pathlib.Path(__file__).parent.absolute())+'//'+gnns[i]+'_'+dset[j]+'_table.tex','w') as f:
                f.write(df2.to_latex(index=False, escape=False, column_format="|lllll"))

    with open(str(pathlib.Path(__file__).parent.absolute())+'//init_table.tex','w') as f:
        f.write(df1.to_latex(index=False, escape=False, column_format="lr"))