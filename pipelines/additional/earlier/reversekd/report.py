"""Progressive results, ROC/PR, per-model plots, Dice distributions and statistics."""
from __future__ import annotations
import html,io,json,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .io import atomic_json,atomic_bytes,read_json,write_csv,file_hash,digest,event,check_pause
from .config import LABELS,GROUPS,counts
from .statistics import KEYS,CONTRASTS,cluster_bootstrap

COLORS={'A':'#578ac5','B':'#4f9b7b','C':'#a96eab','D':'#dd9357','E':'#4eafbb','F':'#cf727b','T':'#4a5567'}
METRICS={'surface_auroc':'Surface AUROC','surface_auprc_ap':'Surface AUPRC (AP)',
 'overlap_auroc':'Localization AUROC','overlap_auprc_ap':'Localization AUPRC (AP)',
 'sensitivity':'Recall / sensitivity, overlap','specificity':'Specificity','precision':'Precision, overlap','mcc':'MCC, overlap',
 'dice_positive_mean':'Dice positive','dice_tp_mean':'Dice TP','jaccard_positive_mean':'Jaccard positive','jaccard_tp_mean':'Jaccard TP',
 'surface_precision':'Surface precision','surface_recall':'Surface recall','f1':'F1, overlap','accuracy':'Accuracy, overlap',
 'pixel_dice_micro':'Pixel Dice (micro)','pixel_jaccard_micro':'Pixel Jaccard (micro)',
 'pixel_precision':'Pixel precision','pixel_recall':'Pixel recall'}
SUMMARY_KEYS=KEYS+['pixel_auroc','pixel_auprc_ap','pixel_auprc_trapezoid']


def collect(c):
    records=[]
    for p in sorted((Path(c['output_dir'])/'evaluations').glob('*/*/*/metrics.json')):records+=read_json(p)
    return pd.DataFrame(records)

def summaries(frame):
    rows=[]
    if frame.empty:return rows
    for (scope,setting,label),g in frame.groupby(['scope','threshold_name','strategy']):
        for key in SUMMARY_KEYS:
            values=pd.to_numeric(g.get(key,pd.Series(np.nan,index=g.index)),errors='coerce');seeds=g.assign(_v=values).groupby('seed')._v.mean().dropna()
            rows.append(dict(scope=scope,threshold_name=setting,strategy=label,metric=key,
                mean=float(seeds.mean()) if len(seeds) else None,
                sd_across_seed_means=float(seeds.std(ddof=1)) if len(seeds)>1 else None,
                sd_across_models=float(values.std(ddof=1)) if values.count()>1 else None,
                n_models=len(g),n_valid_models=int(values.count()),n_valid_seeds=len(seeds),
                n_dice_tp=int(g.n_dice_tp.sum()),n_dice_positive=int(g.n_dice_positive.sum()),
                threshold_min=float(g.threshold.min()),threshold_max=float(g.threshold.max())))
    return rows

def fmt(value):return 'NA' if value is None or pd.isna(value) else f'{float(value):.3f}'

def refresh(c,plots=False):
    out=Path(c['output_dir']);rep=out/'reports';rep.mkdir(parents=True,exist_ok=True)
    frame=collect(c);long=summaries(frame)
    stages=[]
    for p in sorted((out/'runs').glob('*/*/result.json')):
        r=read_json(p);s=r['spec'];m=r.get('metrics') or {}
        stages.append(dict(task_id=s['task_id'],seed=s['seed'],fold=s['fold'],stage=s['stage'],domain=s['domain'],
            kd=s['kd'],selection=s['selection'],epochs_completed=r['epochs_completed'],best_epoch=r['best_epoch'],
            checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],**{k:m.get(k) for k in KEYS}))
    write_csv(rep/'training_stages.csv',stages)
    if not frame.empty:write_csv(rep/'per_model_metrics.csv',frame.to_dict('records'))
    write_csv(rep/'mean_sd_long.csv',long)
    lines=['ACTA REVERSE KNOWLEDGE DISTILLATION','',
      'Professor: frozen YOLO26 semantic trained on ACTA crops. Student: plan-defined nnU-Net v2 2D.',
      'Full-frame custom Dice+CE training. ROI-only soft KL in the DCBR stage of C/D/F.',
      'Same original DCBR folds. ACTA is the external source pool; no source tooth is used as a clinical test case.',
      f'HPO blocks: {c["hpo_enabled"] or "OFF"}. Final fits completed: {len(stages)}/{counts(c)["training_stages_total"]}.',
      'A=N. D=K. B starts from A. C starts from D. E/F start from the same ACTA-pretrained V.',
      'A/D shared initial clinical fits are reused; no extra refit just to rename a checkpoint.',
      'Optimizers reset at each stage. A/D share random initialization and clinical minibatch ordering.',
      'Checkpoint selection: inner-validation fixed epochs when HPO is enabled; otherwise clinical validation in A/B/C/D/E/F, fixed last ACTA source V.',
      'T is evaluated using supplied anatomical proximal ROIs; A-F receive the full bitewing directly.',
      'T comparisons describe different input pipelines. They do not isolate architecture alone.',
      'B-E compares complete training orders/protocols, including stage learning rates and checkpoint selection.',
      '',
      'Mean: folds averaged within seed, then seeds. SD of seed means and model-level SD saved separately.',
      'Seeds/folds are repeated fitted models, not independent patients. Conditional CIs use paired patient/bitewing cluster bootstrap.',
      'HPO uses only a grouped inner holdout of the original training fold. Final epochs and Youden freeze before outer-fold evaluation. Without HPO, original DCBR validation selects checkpoints/Youden and is a development endpoint.',
      'Reference25 was already inspected. It is a secondary evaluation, not prospective independent confirmation.',
      'Dice-positive includes missed lesions as zero. Dice-TP includes only overlapping true positives; NA if there are none.',
      'AUROC/AUPRC use continuous scores. AP and trapezoidal AUPRC are saved separately. Thresholds: fixed 0.5 and DCBR-only Youden. Exact pooled pixel areas are point estimates, without pixel-curve bootstrap CIs.',
      'When patient identifiers are absent, grouping is by bitewing and patient independence cannot be verified.',
      'Partial reports identify actual evaluated model counts. Do not compare unequal completion as final results.','']
    for scope in ('development_validation','reference25'):
        for setting in ('fixed','youden'):
            lines+=[scope.upper()+' | '+setting.upper(), 'Group n  AUROC     AP(S)     AP(L)     Sens      Spec      MCC       Dice+     Dice-TP     Jaccard+  Jaccard-TP  Precision Recall']
            for label in GROUPS:
                rr={r['metric']:r for r in long if (r['scope'],r['threshold_name'],r['strategy'])==(scope,setting,label)}
                if not rr:continue
                first=next(iter(rr.values()))
                lines.append(f'{label:<5} {first["n_models"]:2} '+ ' '.join(f'{fmt(rr[k]["mean"]):<9}' for k in ('surface_auroc','surface_auprc_ap','overlap_auprc_ap','sensitivity','specificity','mcc','dice_positive_mean','dice_tp_mean','jaccard_positive_mean','jaccard_tp_mean','precision','recall')))
                if setting=='youden':lines.append(f'       thresholds {first["threshold_min"]:.6g} to {first["threshold_max"]:.6g}')
            lines.append('')
    monitor_lines=lines.copy()
    lines+=['ALL METRICS (mean, model SD and seed SD)',pd.DataFrame(long).to_string(index=False) if long else 'No evaluated models.', '', 'PLANNED CONTRASTS: '+', '.join(a+' - '+b for a,b in CONTRASTS)]
    stats=rep/'statistics_status.json'
    if stats.is_file():lines+=['','STATISTICS STATUS',json.dumps(read_json(stats),indent=2)]
    statistical_files=list(rep.glob('*/*_bootstrap_ci.csv'))+list(rep.glob('*/*_paired_differences.csv'))+list(rep.glob('*/glmm/*/planned_contrasts.csv'))
    details=[]
    for p in statistical_files:
        details+=['',str(p.relative_to(rep)),pd.read_csv(p).to_string(index=False)]
    text='\n'.join(lines)+'\n'
    atomic_bytes(rep/'MONITOR.txt',('\n'.join(monitor_lines)+'\n').encode());atomic_bytes(rep/'RESULTS.txt',(text+'\n'.join(details)).encode())
    atomic_bytes(rep/'STATISTICS.txt',('\n'.join(details) or 'Statistics not computed yet.').encode())
    if plots and not frame.empty:make_plots(c,frame)
    if plots:learning_curves(c)
    pictures=''.join(f'<figure><img src="{p.relative_to(rep)}" style="max-width:100%"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in sorted(rep.glob('*/plots/*.png')))
    atomic_bytes(rep/'index.html',('<!doctype html><html><meta charset="utf-8"><title>ACTA reverse KD</title>'
      '<style>body{font-family:Arial;max-width:1400px;margin:30px auto}pre{overflow:auto}figure{margin:30px 0}</style>'
      '<h1>ACTA reverse KD</h1><pre>'+html.escape(text)+'</pre>'+pictures+'</html>').encode())
    if c['hpo_enabled']:
        from .hpo import write_progress
        write_progress(c)
        if plots:hpo_plots(c)
    return frame

def savefig(fig,p):
    import matplotlib.pyplot as plt
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        b=io.BytesIO();fig.savefig(b,format=extension,dpi=150,bbox_inches='tight');atomic_bytes(p.with_suffix('.'+extension),b.getvalue())
    plt.close(fig)

def make_plots(c,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    try:
        import seaborn as sns
        sns.set_theme(style='whitegrid',context='notebook')
    except ImportError:sns=None
    out=Path(c['output_dir'])
    for scope in frame.scope.unique():
        folder=out/'reports'/scope/'plots';f=frame[frame.scope==scope]
        labels=[x for x in GROUPS if x in set(f.strategy)]
        parts=[pd.read_csv(p) for p in (out/'evaluations'/scope).glob('*/*/surface_results.csv')]
        d=pd.concat(parts,ignore_index=True);fixed=d[d.threshold_name=='fixed']
        for kind in ('roc','pr','localization_roc','localization_pr'):
            fig,ax=plt.subplots(figsize=(8,6));grid=np.linspace(0,1,301)
            for label in labels:
                curves=[];areas=[]
                for _,g in fixed[fixed.strategy==label].groupby(['seed','fold']):
                    y=g.reference_positive.to_numpy();score=np.where(g.reference_positive.to_numpy()==1,g.overlap_score.fillna(0).to_numpy(),g.score.to_numpy()) if kind.startswith('localization') else g.score.to_numpy()
                    if len(np.unique(y))<2:continue
                    if kind.endswith('roc'):
                        x,z,_=roc_curve(y,score);area=roc_auc_score(y,score)
                        # Collapse duplicate FPR coordinates by maximum sensitivity.
                        unique=np.unique(x);z=np.array([z[x==u].max() for u in unique]);x=unique
                    else:
                        z,x,_=precision_recall_curve(y,score);area=average_precision_score(y,score)
                        order=np.argsort(x,kind='stable');x=x[order];z=z[order]
                        unique=np.unique(x);z=np.array([z[x==u].max() for u in unique]);x=unique
                    curves.append(np.interp(grid,x,z));areas.append(area)
                if not curves:continue
                mean=np.mean(curves,axis=0);sd=np.std(curves,axis=0,ddof=1) if len(curves)>1 else np.zeros_like(mean)
                ax.plot(grid,mean,color=COLORS[label],label=f'{label}: {np.mean(areas):.3f} ± {np.std(areas,ddof=1) if len(areas)>1 else 0:.3f}, n={len(areas)}')
                ax.fill_between(grid,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=COLORS[label],alpha=.08)
            if kind.endswith('roc'):ax.plot([0,1],[0,1],color='grey',ls='--');ax.set(xlabel='False-positive rate',ylabel='Sensitivity')
            else:
                baseline=fixed.drop_duplicates('sample_id').reference_positive.mean()
                ax.axhline(baseline,color='grey',ls='--');ax.set(xlabel='Recall',ylabel='Precision')
            ax.set(xlim=(0,1),ylim=(0,1.02),title=f'{scope}: surface {kind.upper()}')
            ax.legend(loc='best',fontsize=9);fig.text(.5,.01,'Mean interpolated curves; areas from original scores. Band/± = model SD, not patient CI.',ha='center',fontsize=8)
            savefig(fig,folder/(kind if kind.startswith('localization') else f'surface_{kind}'))
        for setting in ('fixed','youden'):
            q=f[f.threshold_name==setting]
            fig,axes=plt.subplots(5,4,figsize=(17,18))
            for ax,(key,title) in zip(axes.flat,METRICS.items()):
                for index,label in enumerate(labels):
                    g=q[q.strategy==label];values=pd.to_numeric(g.get(key,pd.Series(np.nan,index=g.index)),errors='coerce').dropna()
                    if len(values):
                        ax.scatter(np.full(len(values),index)+np.linspace(-.10,.10,len(values)),values,color=COLORS[label],alpha=.65,s=20)
                        seed_means=g.assign(_v=pd.to_numeric(g.get(key,pd.Series(np.nan,index=g.index)),errors='coerce')).groupby('seed')._v.mean().dropna()
                        ax.plot(index,seed_means.mean(),'_',color='black',ms=17,mew=2)
                ax.set(xticks=range(len(labels)),xticklabels=labels,title=title,ylim=(-1.05,1.05) if key=='mcc' else (0,1.05))
            fig.suptitle(f'{scope}, {setting}: dots = fitted seed/fold models; black mark = mean',fontsize=14)
            fig.tight_layout(rect=(0,0,1,.96));savefig(fig,folder/f'{setting}_per_model_metrics')
            fig,axes=plt.subplots(1,2,figsize=(13,5));u=d[d.threshold_name==setting]
            for ax,(name,eligible,title) in zip(axes,[('dice_positive',u.reference_positive==1,'Dice-positive: all diseased surfaces'),
                ('dice_positive',(u.reference_positive==1)&(u.intersection_pixels>0),'Dice-TP: overlapping true positives')]):
                z=u.loc[eligible].copy()
                if not z.empty and sns is not None:
                    sns.violinplot(data=z,x='strategy',y=name,order=labels,palette=COLORS,hue='strategy',
                                   hue_order=labels,dodge=False,inner='box',cut=0,ax=ax)
                    if ax.get_legend() is not None:ax.get_legend().remove()
                elif not z.empty:
                    data=[z.loc[z.strategy==label,name].dropna().to_numpy() for label in labels]
                    nonempty=[(i,a) for i,a in enumerate(data) if len(a)]
                    if nonempty:ax.boxplot([a for _,a in nonempty],positions=[i for i,_ in nonempty],widths=.5)
                ax.set(title=title,ylim=(-.02,1.02),xlabel='Strategy',ylabel='Dice',xticks=range(len(labels)),xticklabels=labels)
                if z.empty:ax.text(.5,.5,'No eligible true positives',transform=ax.transAxes,ha='center')
            fig.suptitle(f'{scope}, {setting}: repeated surface × model scores, not independent patients',fontsize=11)
            fig.tight_layout();savefig(fig,folder/f'{setting}_dice_distributions')
            jaccard_distributions(c,u,labels,folder,scope,setting)
            confusion_plots(q,labels,folder,scope,setting)
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            for ax,key in zip(axes,('surface_auroc','dice_positive_mean')):
                for label in labels:
                    g=q[q.strategy==label].groupby('fold')[key].mean()
                    ax.plot(g.index,g.values,'o-',color=COLORS[label],label=label)
                ax.set(title=METRICS[key],xlabel='Original fold',ylabel='Mean across seeds',ylim=(0,1.02),xticks=range(c['folds']))
            axes[-1].legend();fig.suptitle(f'{scope}, {setting}: original folds');fig.tight_layout()
            savefig(fig,folder/f'{setting}_per_fold')

def learning_curves(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(c['output_dir']);rows=[]
    for p in sorted((out/'runs').glob('*/*/epochs/*.json')):rows.append(read_json(p))
    if not rows:return
    frame=pd.DataFrame(rows);write_csv(out/'reports/training_history.csv',frame.to_dict('records'))
    for stage,g in frame.groupby('stage'):
        fig,axes=plt.subplots(3,5,figsize=(16,8),squeeze=False)
        for (seed,fold),part in g.groupby(['seed','fold']):
            ax=axes[c['seeds'].index(seed),int(fold)];part=part.sort_values('epoch')
            for key,label in [('loss','Total'),('supervised_loss','Dice+CE'),('distillation_loss','KL')]:
                ax.plot(part.epoch,part[key],label=label,lw=1)
            ax.set(title=f'seed {seed}, fold {fold}',xlabel='Epoch',ylabel='Loss')
        axes[0,0].legend(fontsize=8);fig.suptitle('Training stage '+stage);fig.tight_layout()
        savefig(fig,out/'reports/training/plots'/f'{stage}_loss')

def finish(c):
    out=Path(c['output_dir']);rep=out/'reports';frame=refresh(c,plots=False);statuses=[]
    expected=len(c['seeds'])*c['folds']*len(GROUPS)*2
    for scope in frame.scope.unique():
        if len(frame[frame.scope==scope])!=expected:
            statuses.append(dict(scope=scope,status='INCOMPLETE_COHORT'));continue
        parts=[pd.read_csv(p) for p in (out/'evaluations'/scope).glob('*/*/surface_results.csv')]
        d=pd.concat(parts,ignore_index=True);folder=rep/scope;write_csv(folder/'all_surface_results.csv',d.to_dict('records'))
        for setting in ('fixed','youden'):
            check_pause(out);receipt=folder/f'{setting}_bootstrap_receipt.json'
            signature=digest(dict(data=file_hash(folder/'all_surface_results.csv'),setting=setting,reporting=c['reporting']))
            outputs=[folder/f'{setting}_bootstrap_ci.csv',folder/f'{setting}_paired_differences.csv']
            if not (receipt.exists() and read_json(receipt)['signature']==signature and all(p.is_file() for p in outputs)):
                cluster_bootstrap(d,c,scope,setting,folder);atomic_json(receipt,dict(signature=signature))
        if c['statistics']['run_glmm']:
            if not shutil.which('Rscript'):
                statuses.append(dict(scope=scope,status='GLMM_PENDING',reason='Rscript is not installed. Training/evaluation completed.'))
            else:
                dest=folder/'glmm';dest.mkdir(exist_ok=True)
                event(out,'glmm_start',scope=scope)
                try:
                    with (dest/'execution.log').open('w') as log:
                        process=subprocess.run(['Rscript',str(Path(c['package_root'])/'analysis/glmm.R'),str(folder/'all_surface_results.csv'),str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=c['statistics']['glmm_timeout_seconds'])
                    statuses.append(dict(scope=scope,status='GLMM_EXECUTED_REVIEW_DIAGNOSTICS' if process.returncode==0 else 'GLMM_PENDING_OR_FAILED',returncode=process.returncode))
                except subprocess.TimeoutExpired:
                    statuses.append(dict(scope=scope,status='GLMM_TIMED_OUT_REVIEW',timeout_seconds=c['statistics']['glmm_timeout_seconds']))
        else:statuses.append(dict(scope=scope,status='BOOTSTRAP_COMPLETE_GLMM_DISABLED'))
    atomic_json(rep/'statistics_status.json',statuses);refresh(c,plots=True)
    return statuses

def export_results(c):
    out=Path(c['output_dir']);target=out/'RESULTS_TO_SHARE.zip';temp=target.with_suffix('.zip.building')
    sources=[p for p in (out/'reports').rglob('*') if p.is_file()]
    sources += [p for p in (out/'audit').glob('*.json') if p.is_file()]
    sources += [p for p in (out/'audit').glob('*.csv') if p.is_file()]
    sources += [p for p in (out/'audit').glob('*.jsonl') if p.is_file()]
    sources += [p for p in (out/'hpo').glob('*/*.json') if p.is_file()]
    sources += [out/'protocol.json'] if (out/'protocol.json').is_file() else []
    for name in ('original_splits_final.json','ORIGINAL_PARTITION_AUDIT.json','split_counts.csv','resolution_audit.csv'):
        p=out/'data'/name
        if p.is_file():sources.append(p)
    models=[]
    for p in sorted((out/'selections').glob('*/*.json')):
        r=read_json(p);models.append(dict(cell=p.parent.name,strategy=p.stem,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256']))
    write_csv(out/'reports/model_index.csv',models)
    if (out/'reports/model_index.csv').is_file() and out/'reports/model_index.csv' not in sources:sources.append(out/'reports/model_index.csv')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sources:z.write(p,str(p.relative_to(out)))
        setup=Path(c['package_root'])/'docs/SETUP.png'
        if setup.is_file():z.write(setup,'SETUP.png')
        z.write(c['config_path'],'configs/study.yaml')
    temp.replace(target);return target


def jaccard_distributions(c,u,labels,folder,scope,setting):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,(eligible,title) in zip(axes,[(u.reference_positive==1,'Jaccard-positive: all diseased surfaces'),
        ((u.reference_positive==1)&(u.intersection_pixels>0),'Jaccard-TP: overlapping true positives')]):
        z=u.loc[eligible]
        values=[pd.to_numeric(z.loc[z.strategy==label,'jaccard_positive'],errors='coerce').dropna().to_numpy() for label in labels]
        nonempty=[(i,a) for i,a in enumerate(values) if len(a)]
        if nonempty:ax.boxplot([a for _,a in nonempty],positions=[i for i,_ in nonempty],widths=.5)
        else:ax.text(.5,.5,'No eligible surfaces',transform=ax.transAxes,ha='center')
        ax.set(title=title,ylim=(-.02,1.02),xlabel='Strategy',ylabel='Jaccard / IoU',xticks=range(len(labels)),xticklabels=labels)
    fig.suptitle(f'{scope}, {setting}: repeated surface × model scores');fig.tight_layout()
    savefig(fig,folder/f'{setting}_jaccard_distributions')

def confusion_plots(q,labels,folder,scope,setting):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,len(labels),figsize=(3*len(labels),9),squeeze=False)
    for row,prefix in enumerate(('','surface_','pixel_')):
        for col,label in enumerate(labels):
            g=q[q.strategy==label];v=[float(g[prefix+k].mean()) for k in ('tn','fp','fn','tp')]
            matrix=np.array(v).reshape(2,2);ax=axes[row,col]
            ax.imshow(matrix,cmap='Blues')
            for y in (0,1):
                for x in (0,1):ax.text(x,y,f'{matrix[y,x]:.1f}',ha='center',va='center')
            ax.set(title=label+' '+(prefix or 'overlap'),xticks=(0,1),yticks=(0,1),xticklabels=('Sound','Caries'),yticklabels=('Sound','Caries'))
            if row==2:ax.set_xlabel('Prediction')
            if col==0:ax.set_ylabel('Reference')
    fig.suptitle(f'{scope}, {setting}: mean confusion counts across fitted models');fig.tight_layout()
    savefig(fig,folder/f'{setting}_confusion_matrices')

def hpo_plots(c):
    import matplotlib.pyplot as plt
    path=Path(c['output_dir'])/'reports/hpo_trials.csv'
    if not path.is_file() or not path.stat().st_size:return
    d=pd.read_csv(path)
    if d.empty:return
    for block,g in d.groupby('block'):
        fig,axes=plt.subplots(3,5,figsize=(17,9),squeeze=False)
        for cell,part in g.groupby('cell'):
            seed=int(cell.split('_')[1]);fold=int(cell.split('_')[3]);ax=axes[c['seeds'].index(seed),fold]
            for label,values in part.groupby('strategy'):ax.plot(values.candidate,values.surface_auroc,'o-',ms=3,label=label,color=COLORS[label])
            ax.set(title=cell,xlabel='HPO candidate',ylabel='Inner AUROC',ylim=(0,1.02))
        axes[0,0].legend(fontsize=8);fig.suptitle('Shared HPO: '+block);fig.tight_layout()
        savefig(fig,Path(c['output_dir'])/'reports/hpo/plots'/block)
