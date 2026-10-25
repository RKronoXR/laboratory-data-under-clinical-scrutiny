"""Read-only cross-profile tables/plots, separate from all within-study selection.

Cross-profile differences are descriptive, paired across the same seed/fold
fits. Their SD is not a patient-sampling confidence interval.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .io import read_json,digest,write_csv,atomic_bytes
from .statistics import KEYS


def compare_available(c):
    base=Path(c['output_base']);destination=base/'COMPARISONS';destination.mkdir(parents=True,exist_ok=True)
    runs=[]
    for folder in sorted(base.iterdir()):
        metrics=folder/'reports/per_model_metrics.csv'
        if folder.is_dir() and metrics.is_file() and (folder/'data/READY.json').is_file():
            ready=read_json(folder/'data/READY.json');bundle=read_json(folder/'data/bundle.json')
            protocol=read_json(folder/'protocol.json')
            config=protocol['config']
            comparable={k:config.get(k) for k in ('student_model','full_image_size','crop_image_size','threshold')}
            key=digest(dict(source=ready['source_zip_sha256'],splits=bundle['splits'],
                            implementation=protocol['code'],versions=protocol['versions'],settings=comparable))
            frame=pd.read_csv(metrics)
            if frame.empty:continue
            runs.append((folder,key,frame))
    combined=[];differences=[];notes=[]
    for folder,key,frame in runs:
        combined.extend(frame.assign(run_id=folder.name,complete=(folder/'COMPLETE.json').exists()).to_dict('records'))
    baseline=[r for r in runs if r[0].name=='NO_HPO']
    if baseline:
        bfolder,bkey,bframe=baseline[0]
        ids=['scope','threshold_name','strategy','seed','fold']
        for folder,key,frame in runs:
            if folder==bfolder:continue
            if key!=bkey:
                notes.append(f'{folder.name}: comparison skipped because data/splits differ.');continue
            pair=bframe.merge(frame,on=ids,suffixes=('_baseline','_hpo'),validate='one_to_one')
            for _,row in pair.iterrows():
                for metric in KEYS:
                    left=row.get(metric+'_baseline');right=row.get(metric+'_hpo')
                    if pd.notna(left) and pd.notna(right):
                        differences.append(dict(run_id=folder.name,baseline=bfolder.name,**{k:row[k] for k in ids},
                            metric=metric,baseline_value=float(left),value=float(right),difference=float(right-left)))
    write_csv(destination/'all_runs_per_model.csv',combined);write_csv(destination/'paired_differences.csv',differences)
    summary=[]
    if differences:
        d=pd.DataFrame(differences)
        for keys,group in d.groupby(['run_id','scope','threshold_name','strategy','metric']):
            seedmeans=group.groupby('seed').difference.mean()
            summary.append(dict(zip(['run_id','scope','threshold_name','strategy','metric'],keys),
                mean_difference=float(seedmeans.mean()),sd_across_seed_means=float(seedmeans.std(ddof=1)) if len(seedmeans)>1 else None,
                n_paired_models=len(group),interpretation='descriptive paired difference; not a patient-level CI'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sd=pd.DataFrame(summary)
        for (run,scope,threshold),g in sd.groupby(['run_id','scope','threshold_name']):
            for metric in ('surface_auroc','surface_auprc_ap','sensitivity','specificity','dice_positive_mean','dice_tp_mean'):
                part=g.loc[g.metric==metric].sort_values('strategy')
                if part.empty:continue
                fig,ax=plt.subplots(figsize=(7,4));ax.axhline(0,linewidth=1,linestyle=':')
                ax.bar(part.strategy,part.mean_difference,yerr=part.sd_across_seed_means.fillna(0),capsize=4)
                ax.set_ylabel('Difference vs NO_HPO (error bar: SD of seed means)')
                ax.set_title(f'{run}\n{scope} | {threshold} | {metric}',fontsize=10)
                fig.tight_layout();p=destination/'plots'/run/scope/threshold;p.mkdir(parents=True,exist_ok=True)
                fig.savefig(p/(metric+'.png'),dpi=160);fig.savefig(p/(metric+'.svg'));plt.close(fig)
    write_csv(destination/'summary_differences.csv',summary)
    message='NO_HPO / HPO COMPARISONS\n\nOnly matching source data and inner/outer splits are paired.\n'
    message+='Each profile keeps its own results, selections and thresholds. Nothing here enters HPO.\n'
    message+='Differences and SD across seed means are descriptive, not patient-level confidence intervals.\n'
    message+='Within each run, patient/bitewing conditional CIs and planned comparisons are available in reports.\n'
    message+='A different optimized teacher makes the comparison an entire-procedure comparison, not an isolated student-HPO effect.\n\n'
    if not differences:message+='No compatible NO_HPO/HPO pair with evaluated models is available yet.\n'
    message+='\n'.join(notes)+'\n'
    atomic_bytes(destination/'COMPARISONS.txt',message.encode())
    return summary
