"""Progressive plain-text/CSV/HTML results and fully labelled scientific plots."""
from __future__ import annotations
import html,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from .io import atomic_bytes,atomic_json,read_json,write_csv,file_hash,digest
from .statistics import KEYS,cluster_bootstrap

LABELS={'T':'T: nnU-Net teacher','A':'A: clinical training','B':'B: ACTA pretraining','C':'C: clinical distillation','D':'D: ACTA + distillation'}
METRIC_LABELS={'surface_auroc':'Surface AUROC','surface_auprc_ap':'Surface AUPRC (average precision)',
'overlap_auprc_ap':'Localization AUPRC (overlap-based AP)','sensitivity':'Sensitivity (overlap required)',
'specificity':'Specificity','mcc':'MCC (overlap endpoint)','dice_positive_mean':'Dice-positive (all reference-positive surfaces)',
'dice_tp_mean':'Dice-TP (overlapping true positives only)','surface_sensitivity':'Sensitivity (surface alarm)',
'surface_specificity':'Specificity (surface alarm)','surface_mcc':'MCC (surface alarm)','accuracy':'Accuracy (overlap endpoint)',
'balanced_accuracy':'Balanced accuracy (overlap endpoint)','ppv':'PPV (overlap endpoint)','npv':'NPV (overlap endpoint)','f1':'F1 (overlap endpoint)'}


def collect(c):
    out=Path(c['output_dir']);records=[]
    for p in sorted((out/'evaluations').glob('*/*/*/metrics.json')):
        records.extend(read_json(p))
    return pd.DataFrame(records)


def summaries(frame):
    rows=[]
    if frame.empty:return rows
    for (scope,setting,strategy),g in frame.groupby(['scope','threshold_name','strategy'],sort=False):
        for key in KEYS:
            values=pd.to_numeric(g[key],errors='coerce')
            seedvalues=g.assign(_v=values).groupby('seed')._v.mean().dropna()
            rows.append(dict(scope=scope,threshold_name=setting,strategy=strategy,metric=key,
                mean=float(seedvalues.mean()) if len(seedvalues) else None,
                sd_across_seed_means=float(seedvalues.std(ddof=1)) if len(seedvalues)>1 else None,
                sd_across_models=float(values.std(ddof=1)) if values.count()>1 else None,
                n_valid_models=int(values.count()),n_models=len(g),n_valid_seeds=len(seedvalues),
                dice_tp_evaluations=int(g.n_dice_tp.sum()),dice_positive_evaluations=int(g.n_dice_positive.sum()),
                threshold_min=float(g.threshold.min()),threshold_max=float(g.threshold.max())))
    return rows


def _fmt(x):return 'NA' if x is None or pd.isna(x) else f'{float(x):.3f}'
def _mean_sd(r):return _fmt(r['mean'])+' +/- '+_fmt(r['sd_across_seed_means'])


def _text(c,frame,long,hpo):
    p=Path(c['output_dir'])/'protocol.json';counts=read_json(p)['counts'] if p.exists() else {}
    text=['ACTA IN VITRO PRETRAINING AND CLINICAL DISTILLATION','',
      'T = nnU-Net v2 2D teacher; A/B/C/D = independently trained YOLO26 semantic models.',
      f'Run profile: {c.get("profile","test")} | HPO enabled: {c.get("hpo_enabled", [])}',
      f'Training fits completed: {sum(not r.get("reused",False) for r in hpo)} / upper bound {counts.get("training_stages_total","not planned yet")}.',
      f'Teachers imported without training: {sum(bool(r.get("reused",False)) for r in hpo)}.',
      f'Completed seed/fold cells: {len(list((Path(c["output_dir"])/"cells").glob("*.json")))} / {counts.get("seed_fold_cells","?")}.',
      'Teacher decisions and any incompatibility: TEACHER_RETRAINING_ALERTS.txt',
      'One stage = one teacher fit, one ACTA pretraining fit, one transfer probe, or one clinical HPO fit.',
      'P is reused by B/D and T is reused by C/D within the same seed/fold. Final weights are NOT shared between A/B/C/D.',
      '', 'MEAN +/- SD: average across folds within each seed, then average across seeds; SD across seed means.',
      'Model-level SD is also saved in mean_sd_long.csv. Neither SD is a patient-level confidence interval.',
      '95% CIs, when available: paired patient/bitewing cluster bootstrap conditional on fitted models.',
      'ROC/AP use continuous scores and do not change with the operating threshold.',
      'AUROC/AP(S): conventional surface detection, using the maximum predicted probability anywhere in each region.',
      'AP(L)/Sens/MCC/Dice-TP: localization endpoint; a diseased surface requires overlap with its reference mask.',
      'Dice-positive includes all diseased surfaces, including missed lesions. Dice-TP excludes missed lesions; NA when no TP.',
      'TP counts are repeated surface-by-model evaluations, not counts of independent lesions.',
      'Youden: '+c['threshold']['youden_endpoint']+' endpoint, selected on the ORIGINAL DCBR validation fold, separately for each fitted model.',
      'No known patient group/bitewing appears in both training and validation within a split.',
      'When patient identifiers are unavailable, patient-level independence cannot be verified; grouping uses bitewings.',
      'Original training/validation folds are preserved. Validation selects checkpoints, HPO and Youden; development scores are not independent test scores.',
      'Reference25 was inspected in earlier work. It is not used for this HPO and is not labelled prospectively blinded.',
      'During partial runs, unequal completion across groups is visible; do not interpret unfinished groups as final comparisons.', '']
    for scope in ('development_validation','reference25'):
        for setting in ('fixed','youden'):
            text.append(f'{scope.upper()} | {setting.upper()}')
            if not long:text.append('No selected models evaluated yet.');continue
            relevant=[r for r in long if r['scope']==scope and r['threshold_name']==setting]
            if not relevant:text.append('No selected models evaluated yet.');continue
            text.append('Group  n  AUROC          AP(S)          AP(L)          Sens           Spec           MCC            Dice+          Dice-TP        TP')
            for label in ('T','A','B','C','D'):
                rr={r['metric']:r for r in relevant if r['strategy']==label}
                if not rr:continue
                names=['surface_auroc','surface_auprc_ap','overlap_auprc_ap','sensitivity','specificity','mcc','dice_positive_mean','dice_tp_mean']
                first=next(iter(rr.values()));v='  '.join(f'{_mean_sd(rr[k]):<13}' for k in names)
                text.append(f'{label:<5} {first["n_models"]:2}  {v}  {first["dice_tp_evaluations"]}')
                if setting=='youden':text.append(f'       Actual threshold range: {first["threshold_min"]:.6g} to {first["threshold_max"]:.6g}')
            text.append('')
    if hpo:
        text.extend(['LATEST COMPLETED TRAINING STAGES (original development validation only; not independent performance)'])
        for r in hpo[-8:]:text.append(f'{r["task_id"]} | AUROC={_fmt(r.get("surface_auroc"))} AP={_fmt(r.get("surface_auprc_ap"))} Sens={_fmt(r.get("sensitivity"))} Spec={_fmt(r.get("specificity"))} MCC={_fmt(r.get("mcc"))} Dice+={_fmt(r.get("dice_positive_mean"))} DiceTP={_fmt(r.get("dice_tp_mean"))} | best_epoch={r.get("best_epoch")}')
    stats=Path(c['output_dir'])/'reports/statistics_status.json'
    if stats.exists():text.extend(['','STATISTICAL ANALYSIS STATUS',json.dumps(read_json(stats),indent=2)])
    text+=['','Detailed outputs: per_model_metrics.csv, mean_sd_long.csv, cohort-specific bootstrap CI/differences, plots and GLMM diagnostics.']
    return '\n'.join(text)+'\n'


def refresh(c,plots=False):
    out=Path(c['output_dir']);rep=out/'reports';rep.mkdir(parents=True,exist_ok=True)
    hpo=[]
    for p in sorted((out/'runs').glob('**/result.json'),key=lambda p:p.stat().st_mtime):
        r=read_json(p);s=r.get('spec',{});m=r.get('metrics') or {}
        hpo.append(dict(task_id=s.get('task_id'),strategy=s.get('strategy'),kind=s.get('kind'),seed=s.get('seed'),fold=s.get('fold'),reused=r.get('reused',False),
            best_epoch=r.get('best_epoch'),epochs_completed=r.get('epochs_completed'),checkpoint_sha256=r.get('checkpoint_sha256'),
            **s.get('hyperparameters',{}),distillation_weight=s.get('distillation_weight',0),**{k:m.get(k) for k in KEYS}))
    write_csv(rep/'hpo_trials.csv',hpo)
    decisions=[]
    for p in sorted((out/'hpo_selections').glob('*/*.json')):
        receipt=read_json(p)
        for r in receipt.get('candidate_audit',[]):
            decisions.append(dict(cell=p.parent.name,phase=p.stem,**r))
    write_csv(rep/'shared_hpo_decisions.csv',decisions)
    frame=collect(c);long=summaries(frame)
    if not frame.empty:write_csv(rep/'per_model_metrics.csv',frame.to_dict('records'))
    write_csv(rep/'mean_sd_long.csv',long)
    text=_text(c,frame,long,hpo);atomic_bytes(rep/'MONITOR.txt',text.encode())
    details=statistics_text(rep)
    atomic_bytes(rep/'STATISTICS.txt',details.encode())
    atomic_bytes(rep/'RESULTS.txt',(text+'\n'+details).encode())
    atomic_bytes(rep/'RESULTS.md',('```text\n'+text+'```\n').encode())
    if plots:
        if not frame.empty:make_plots(c,frame,long)
        additional_plots(c,frame,hpo)
    links=[]
    for p in sorted(rep.glob('**/plots/*.png')):links.append(f'<figure><img loading="lazy" src="{p.relative_to(rep)}" style="max-width:100%"><figcaption>{html.escape(p.stem)}</figcaption></figure>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>ACTA model comparison</title><style>body{font-family:Arial;max-width:1500px;margin:30px auto}pre{white-space:pre;overflow:auto;font-size:13px}figure{max-width:1000px}</style><h1>ACTA model comparison</h1><pre>'+html.escape(text)+'</pre>'+''.join(links)+'</html>'
    atomic_bytes(rep/'index.html',page.encode())
    return frame


def _save(fig,p):
    import matplotlib.pyplot as plt
    import io
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    for fmt,target in [('png',p),('svg',p.with_suffix('.svg'))]:
        buffer=io.BytesIO();fig.savefig(buffer,format=fmt,dpi=160,bbox_inches='tight');atomic_bytes(target,buffer.getvalue())
    plt.close(fig)


def make_plots(c,frame,long):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(c['output_dir']);rep=out/'reports'
    for scope in frame.scope.unique():
        folder=rep/scope/'plots';f=frame.loc[frame.scope==scope]
        cache=rep/scope/'base_plot_cache.json';signature=digest(f.to_dict('records'))
        if cache.exists() and read_json(cache).get('signature')==signature:continue
        labels=[k for k in ('T','A','B','C','D') if k in set(f.strategy)]
        surfaces=[]
        for p in (out/'evaluations'/scope).glob('*/*/surface_results.csv'):surfaces.append(pd.read_csv(p))
        d=pd.concat(surfaces,ignore_index=True)
        for mode in ('surface','overlap'):
            for kind in ('roc','precision_recall'):
                fig,ax=plt.subplots(figsize=(8,6));grid=np.linspace(0,1,301)
                for label in labels:
                    curves=[]
                    for p in (out/'evaluations'/scope).glob(f'*/{label}/fixed_{mode}_curve.csv'):
                        q=pd.read_csv(p);x=q.fpr.to_numpy() if kind=='roc' else q.recall.to_numpy();y=q.recall.to_numpy() if kind=='roc' else q.precision.to_numpy()
                        # Keep the last threshold point for duplicate x; no monotone precision envelope.
                        values=pd.DataFrame({'x':x,'y':y}).groupby('x').y.last()
                        curves.append(np.interp(grid,values.index,values.values))
                    if not curves:continue
                    metric=f'{mode}_auroc' if kind=='roc' else f'{mode}_auprc_ap'
                    vals=f.loc[(f.strategy==label)&(f.threshold_name=='fixed'),metric].dropna()
                    sd=vals.std(ddof=1) if len(vals)>1 else np.nan
                    ax.plot(grid,np.mean(curves,axis=0),label=f'{label}: mean area={vals.mean():.3f}, model SD={_fmt(sd)}, n={len(vals)}')
                if kind=='roc':ax.plot([0,1],[0,1],linestyle='--',label='No-discrimination line');ax.set_xlabel('False-positive rate (1 - specificity)');ax.set_ylabel('True-positive rate (sensitivity)')
                else:
                    base=d.loc[d.threshold_name=='fixed'].groupby('model_id').reference_positive.mean().mean()
                    ax.axhline(base,linestyle='--',label=f'Mean positive-surface prevalence = {base:.3f}');ax.set_xlabel('Recall (sensitivity)');ax.set_ylabel('Precision (positive predictive value)')
                ax.set(xlim=(0,1),ylim=(0,1.02));ax.set_title(f'{scope}: {mode} {kind.replace("_"," ")}\nMean interpolated curves; areas calculated on original curves, not on this mean')
                ax.legend(fontsize=8,loc='best');ax.grid(alpha=.2);_save(fig,folder/f'{mode}_{kind}.png')
        for key in ['surface_auroc','surface_auprc_ap','sensitivity','specificity','mcc','dice_positive_mean','dice_tp_mean']:
            fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(labels))
            for j,setting in enumerate(('fixed','youden')):
                rr={(r['strategy']):r for r in long if r['scope']==scope and r['threshold_name']==setting and r['metric']==key}
                vals=[rr[k]['mean'] if k in rr and rr[k]['mean'] is not None else np.nan for k in labels]
                # Descriptive SD, explicitly distinct from confidence intervals.
                errs=[rr[k]['sd_across_seed_means'] if k in rr and rr[k]['sd_across_seed_means'] is not None else 0 for k in labels]
                ax.errorbar(x+(j-.5)*.18,vals,yerr=errs,fmt='o' if j==0 else 's',capsize=4,label='Fixed 0.5' if j==0 else 'Development-validation Youden (model-specific)')
            ax.set_xticks(x,labels);ax.set_xlabel('Independently trained model group');ax.set_ylabel(METRIC_LABELS[key]);ax.set_title(f'{scope}\nMean +/- SD across seed means; zero-length bars when SD is not estimable')
            ax.set_ylim((-1,1) if key=='mcc' else (0,1));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);_save(fig,folder/f'{key}_mean_sd.png')
        for setting in ('fixed','youden'):
            q=d.loc[d.threshold_name==setting]
            for tp_only in (False,True):
                part=q.loc[(q.reference_positive==1)&((q.intersection_pixels>0) if tp_only else True)]
                series=[part.loc[part.strategy==k,'dice_positive'].dropna().to_numpy() for k in labels]
                fig,ax=plt.subplots(figsize=(9,6));usable=[(i+1,a) for i,a in enumerate(series) if len(a)]
                for position,a in usable:
                    if len(a)>1 and np.ptp(a)>0:ax.violinplot([a],positions=[position],showmedians=True,showextrema=False)
                    ax.boxplot([a],positions=[position],widths=.16,showfliers=False,manage_ticks=False)
                ax.set_xticks(np.arange(1,len(labels)+1),[f'{k}\nn={len(a)}' for k,a in zip(labels,series)])
                ax.set(ylim=(0,1),ylabel='Dice score',xlabel='Model group; n = surface-by-model evaluations (dependent repetitions)')
                ax.set_title(f'{scope} | {setting}\n'+('Dice-TP: only overlapping detected positives' if tp_only else 'Dice-positive: all reference-positive surfaces, including misses'))
                ax.text(.5,-.23,'Violin: density; box: interquartile range; line: median; whiskers: 1.5 IQR. Outliers hidden.',transform=ax.transAxes,ha='center',fontsize=8)
                ax.grid(axis='y',alpha=.2);_save(fig,folder/f'{setting}_dice_{"tp" if tp_only else "positive"}_distribution.png')
            for label in labels:
                m=f.loc[(f.strategy==label)&(f.threshold_name==setting)]
                matrix=np.array([[m.tn.sum(),m.fp.sum()],[m.fn.sum(),m.tp.sum()]],dtype=int)
                fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(matrix,alpha=.25)
                for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:,}',ha='center',va='center',fontsize=14)
                ax.set_xticks([0,1],['Negative','Positive']);ax.set_yticks([0,1],['Sound','Lesion']);ax.set_xlabel('Localization decision');ax.set_ylabel('Reference standard')
                ax.set_title(f'{scope} | {label} | {setting}\nCounts pooled over {len(m)} fitted models; repeated evaluations\nDarker shading represents larger counts')
                _save(fig,folder/f'{setting}_{label}_confusion.png')
        fig,ax=plt.subplots(figsize=(8,5))
        for i,label in enumerate(labels):
            vals=f.loc[(f.strategy==label)&(f.threshold_name=='youden'),'threshold'].values
            ax.plot(np.full(len(vals),i),vals,'o',alpha=.6)
        ax.axhline(.5,linestyle='--',label='Fixed threshold 0.5');ax.set_xticks(range(len(labels)),labels)
        ax.set(ylim=(0,1.02),ylabel='Probability threshold',xlabel='Model group',title=f'{scope}: development-validation Youden thresholds\nEach point is one fitted model; thresholds fitted on DCBR validation, never on Reference25')
        ax.legend();_save(fig,folder/'youden_thresholds.png')
        atomic_json(cache,{'signature':signature})


def finish(c):
    out=Path(c['output_dir']);rep=out/'reports';frame=refresh(c,plots=False)
    statuses=[]
    if frame.empty:raise RuntimeError('No evaluated models available for final statistical analysis.')
    expected=len(c['seeds'])*c['folds']*5*2
    for scope in frame.scope.unique():
        if len(frame.loc[frame.scope==scope])!=expected:
            statuses.append(dict(scope=scope,status='WAITING_FOR_COMPLETE_COHORT',expected_metric_rows=expected))
            continue
        parts=[pd.read_csv(p) for p in (out/'evaluations'/scope).glob('*/*/surface_results.csv')]
        d=pd.concat(parts,ignore_index=True);folder=rep/scope;write_csv(folder/'all_surface_results.csv',d.to_dict('records'))
        for setting in ('fixed','youden'):
            receipt=folder/f'{setting}_bootstrap_receipt.json'
            signature=digest(dict(input=file_hash(folder/'all_surface_results.csv'),reporting=c['reporting'],setting=setting))
            outputs=[folder/f'{setting}_bootstrap_ci.csv',folder/f'{setting}_paired_differences.csv']
            valid=receipt.exists() and read_json(receipt).get('signature')==signature and all(p.is_file() for p in outputs)
            if not valid:
                cluster_bootstrap(d,c,scope,setting,folder)
                atomic_json(receipt,dict(signature=signature))
        if c['statistics']['run_glmm']:
            import shutil
            if not shutil.which('Rscript'):
                statuses.append(dict(scope=scope,status='NOT_RUN',reason='Rscript is missing. Run INSTALL_STATISTICS.sh, then REPORT.sh --final.'))
            else:
                target=folder/'glmm';target.mkdir(exist_ok=True)
                with (target/'execution.log').open('w') as log:
                    run=subprocess.run(['Rscript',str(Path(c['package_root'])/'analysis/glmm.R'),str(folder/'all_surface_results.csv'),str(target)],stdout=log,stderr=subprocess.STDOUT)
                statuses.append(dict(scope=scope,status='EXECUTED_REVIEW_DIAGNOSTICS' if run.returncode==0 else 'FAILED',returncode=run.returncode,log=str(target/'execution.log')))
    atomic_json(rep/'statistics_status.json',statuses)
    refresh(c,plots=True)


def statistics_text(rep):
    """Every CI/contrast is available in one readable text file, not just CSV."""
    lines=['STATISTICAL RESULTS','95% cluster CIs are conditional on fitted models; pointwise, not simultaneous.',
           'DCBR DEVELOPMENT: exploratory, selection-conditional. CI/regression does not correct selection bias.',
           'Reference25: evaluation-only here, previously inspected; not prospectively blinded.','']
    found=False
    for scope in ('development_validation','reference25'):
        for setting in ('fixed','youden'):
            for suffix in ('bootstrap_ci','paired_differences'):
                p=Path(rep)/scope/f'{setting}_{suffix}.csv'
                if not p.is_file():continue
                found=True;frame=pd.read_csv(p)
                lines.extend([f'{scope} | {setting} | {suffix}',frame.to_string(index=False),''])
        for p in sorted((Path(rep)/scope/'glmm').glob('*/planned_contrasts.csv')):
            found=True;lines.extend([str(p.relative_to(rep)),pd.read_csv(p).to_string(index=False),''])
        for p in sorted((Path(rep)/scope/'glmm').glob('*/diagnostics.json')):
            lines.extend([str(p.relative_to(rep)),json.dumps(read_json(p),indent=2),''])
    if not found:lines.append('Not available yet. Final statistical inference runs after all development selections are frozen.')
    return '\n'.join(lines)+'\n'


def additional_plots(c,frame,hpo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(c['output_dir']);rep=out/'reports'
    # HPO traces show original development validation, not independent test performance.
    hcache=rep/'hpo/plot_cache.json';hsig=digest(hpo)
    if hpo and (not hcache.exists() or read_json(hcache).get('signature')!=hsig):
        h=pd.DataFrame(hpo)
        for label in ('T','A','P_probe','B','C','D'):
            sub=h.loc[h.strategy==label].copy()
            if sub.empty:continue
            fig,ax=plt.subplots(figsize=(10,6))
            for (seed,fold),g in sub.groupby(['seed','fold']):
                trial=pd.to_numeric(g.task_id.str.extract(r'trial_(\d+)')[0],errors='coerce').fillna(0).astype(int).to_numpy()+1
                styles=['-','--',':','-.'];style=styles[sorted(sub.seed.unique()).index(seed)%len(styles)]
                ax.plot(trial,pd.to_numeric(g.surface_auroc,errors='coerce'),marker='o',linestyle=style,markersize=3,linewidth=.7,label=f'Seed {seed}, fold {fold}')
            ax.set(xlabel='HPO configuration number',ylabel='Development-validation surface AUROC',ylim=(0,1),
                title=f'{label}: hyperparameter search\nSelection-data results, not independent test performance')
            ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2);_save(fig,rep/'hpo/plots'/f'{label}_search.png')
        atomic_json(hcache,{'signature':hsig})
    if not frame.empty:
        for scope in frame.scope.unique():
            cache=rep/scope/'additional_plot_cache.json'
            sig=digest({'models':frame.loc[frame.scope==scope].to_dict('records'),
                'ci_files':{p.name:file_hash(p) for p in (rep/scope).glob('*_paired_differences.csv')}})
            if cache.exists() and read_json(cache).get('signature')==sig:continue
            for setting in ('fixed','youden'):
                q=frame.loc[(frame.scope==scope)&(frame.threshold_name==setting)]
                cells=sorted({(int(r.seed),int(r.fold)) for r in q.itertuples()},key=lambda a:(a[1],a[0]))
                for key in ('surface_auroc','surface_auprc_ap','sensitivity','specificity','mcc','dice_positive_mean','dice_tp_mean'):
                    fig,ax=plt.subplots(figsize=(11,5))
                    for label in ('T','A','B','C','D'):
                        g=q.loc[q.strategy==label]
                        if g.empty:continue
                        by={(int(r.seed),int(r.fold)):getattr(r,key) for r in g.itertuples()}
                        ax.plot(range(len(cells)),[by.get(pair,np.nan) for pair in cells],'o-',label=label,markersize=4)
                    ax.set_xticks(range(len(cells)),[f'S{s}\nF{f}' for s,f in cells],fontsize=8)
                    ax.set(xlabel='One fitted model per seed (S) and original fold (F)',ylabel=METRIC_LABELS[key],
                           title=f'{scope} | {setting}: per-model results',ylim=(-1,1) if key=='mcc' else (0,1))
                    ax.legend();ax.grid(alpha=.2);_save(fig,rep/scope/'plots'/f'{setting}_{key}_per_model.png')
                # Paired-difference forest plots, generated when final CIs exist.
                p=rep/scope/f'{setting}_paired_differences.csv'
                if not p.is_file():continue
                ds=pd.read_csv(p)
                for key in ('surface_auroc','surface_auprc_ap','sensitivity','specificity','mcc','dice_positive_mean','dice_tp_mean'):
                    g=ds.loc[ds.metric==key].reset_index(drop=True);fig,ax=plt.subplots(figsize=(9,5))
                    valid=g.ci_low.notna()&g.ci_high.notna()
                    ax.hlines(g.index[valid],g.loc[valid,'ci_low'],g.loc[valid,'ci_high'],linewidth=2)
                    ax.plot(g.difference,g.index,'o',label='Point estimate')
                    ax.axvline(0,linestyle='--',label='No difference');ax.set_yticks(range(len(g)),g.comparison)
                    ax.set(xlabel=f'Difference in {METRIC_LABELS[key]} (absolute units)',title=f'{scope} | {setting}\nPaired cluster 95% CIs, conditional on fitted models; pointwise')
                    ax.legend();ax.grid(axis='x',alpha=.2);_save(fig,rep/scope/'plots'/f'{setting}_{key}_paired_ci.png')
            atomic_json(cache,{'signature':sig})
    # Selected-model learning histories. All candidates keep their raw epoch logs.
    for selection in sorted((out/'selections').glob('*/*.json')):
        r=read_json(selection);task=r['spec'];files=sorted((Path(task['directory'])/'epochs').glob('*.json'))
        if not files:continue
        label=selection.stem;cell=selection.parent.name
        picture=rep/'learning/plots'/f'{label}_{cell}_loss.png'
        if picture.exists() and picture.with_suffix('.svg').exists():continue
        history=[read_json(p) for p in files]
        fig,ax=plt.subplots(figsize=(9,5))
        if label=='T':
            pairs=[(x.get('epoch'),x.get('metrics',{}).get('train_loss'),x.get('metrics',{}).get('validation_loss')) for x in history]
            ax.plot([a[0] for a in pairs],[a[1] for a in pairs],label='Training loss')
            ax.plot([a[0] for a in pairs],[a[2] for a in pairs],label='Native validation loss')
        else:
            for field,description in [('loss','Total loss'),('supervised_loss','Supervised loss'),('distillation_loss','Distillation KL loss (unweighted)')]:
                if field=='distillation_loss' and label not in ('C','D'):continue
                ax.plot([x['epoch'] for x in history],[x.get(field,np.nan) for x in history],label=description)
        if r.get('best_epoch') is not None:ax.axvline(r['best_epoch'],linestyle='--',label=f'Selected epoch {r["best_epoch"]}')
        ax.set(xlabel='Epoch',ylabel='Loss',title=f'{label} | {cell}: selected model training history')
        ax.legend(fontsize=8);ax.grid(alpha=.2);_save(fig,rep/'learning/plots'/f'{label}_{cell}_loss.png')
