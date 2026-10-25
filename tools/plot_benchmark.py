#!/usr/bin/env python3
"""Plot the recorded architecture benchmark using seed means and sample SD."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def values(rows,group,metric):
    selected=[r for r in rows if r['group_id']==group and r['threshold_name']=='fixed']
    if {int(r['seed']) for r in selected}!={17,42,2026} or len(selected)!=3:
        raise ValueError('Expected the three complete recorded seed summaries: '+group)
    v=[float(r[metric]) for r in selected]
    return statistics.mean(v),statistics.stdev(v)


def plot(output,dpi=600):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator,FormatStrFormatter
    import numpy as np
    native=read(REPO/'data/benchmark_reference/per_seed_native.csv')
    paired=read(REPO/'data/benchmark_reference/per_seed_paired.csv')
    panels=[(native,'acta_direct_crops','(a) ACTA-DIRECT\nProximal-side region crops'),
            (paired,'dcbr_full','(b) Clinical images\nBitewings'),
            (paired,'dcbr_crops','(c) Clinical images\nProximal-side region crops')]
    architectures=[('nnunetv2','nnU-Net','#0072B2'),('yolo26_sem','YOLO','#D55E00'),('monai_segresnet','SegResNet','#7F7F7F')]
    charts=[('Figure_5_4_architecture_detection',[('sensitivity','Sensitivity'),('specificity','Specificity')],'Detection proportion',1.05,.25),
            ('Figure_5_5_architecture_segmentation',[('dice_positive_mean','Dice-Positive'),('dice_tp_mean','Dice-TP')],'Dice similarity',.68,.2)]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}):
        for name,metrics,ylabel,upper,tick in charts:
            fig,axes=plt.subplots(1,3,figsize=(11.0,4.4),sharey=True)
            x=np.arange(len(metrics));width=.23
            for ax,(rows,suffix,title) in zip(axes,panels):
                for i,(key,label,color) in enumerate(architectures):
                    summary=[values(rows,key+'_'+suffix,m) for m,_ in metrics]
                    ax.bar(x+(i-1)*width,[a for a,b in summary],width,yerr=[b for a,b in summary],label=label,color=color,
                           error_kw={'elinewidth':1,'capsize':3,'capthick':1},zorder=3)
                ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=12)
                ax.set_xticks(x,[label for _,label in metrics]);ax.set_ylim(0,upper)
                ax.yaxis.set_major_locator(MultipleLocator(tick));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if name.startswith('Figure_5_4') else '%.1f'))
                ax.grid(axis='y',color='#DFE6EE',linewidth=.8,zorder=0);ax.set_axisbelow(True)
            axes[0].set_ylabel(ylabel,fontsize=11)
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=11)
            fig.subplots_adjust(left=.075,right=.99,bottom=.14,top=.78,wspace=.19)
            for extension in ['png','svg']:
                fig.savefig(output/(name+'.'+extension),dpi=dpi,bbox_inches='tight',facecolor='white')
            plt.close(fig)
    return [str(output/(name+'.'+ext)) for name,*_ in charts for ext in ['png','svg']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=REPO/'results/figures')
    p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    if args.dpi<72:p.error('--dpi must be at least 72')
    for path in plot(args.output_dir,args.dpi):print(path)


if __name__=='__main__':main()
