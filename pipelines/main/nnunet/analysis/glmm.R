#!/usr/bin/env Rscript
# Binary localization correctness, sensitivity and specificity only.
# AUROC/AP/Dice/MCC contrasts are evaluated by the paired cluster bootstrap.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) stop('Usage: Rscript analysis/glmm.R surface_results.csv output_dir')
needed <- c('lme4','emmeans','jsonlite')
if (any(!vapply(needed,requireNamespace,logical(1),quietly=TRUE))) stop('Missing R packages. Run INSTALL_STATISTICS.sh')
out <- args[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
d <- read.csv(args[1],stringsAsFactors=FALSE)
levels <- c('A','B','C','D','E','F','T','D0','C0')
keys<-c('threshold_name','strategy','sample_id','seed','fold')
if(anyDuplicated(d[keys])) stop('Duplicate paired surface/model repetitions are not allowed.')
if(any(!d$strategy %in% levels)) stop('Unknown strategy label.')
for(field in intersect(c('group_id','bitewing_id','patient_id','reference_positive','reference_pixels'),names(d))) {
 aligned<-tapply(d[[field]],d$sample_id,function(x)length(unique(x)))
 if(any(aligned>1)) stop(paste('Paired surface reference/cluster alignment differs for',field))
}
contrasts <- list()
for (pair in combn(levels,2,simplify=FALSE)) {
 if(identical(pair,c('E','F'))) pair<-rev(pair)
 v<-setNames(rep(0,length(levels)),levels);v[pair[1]]<-1;v[pair[2]]<--1
 contrasts[[paste(pair,collapse=' - ')]]<-unname(v)
}
for (setting in unique(d$threshold_name)) {
 for (endpoint in c('correctness','sensitivity','specificity')) {
  dest <- file.path(out,paste(setting,endpoint,sep='_'));dir.create(dest,recursive=TRUE,showWarnings=FALSE)
  tryCatch({
   a <- d[d$threshold_name==setting,]
   if (endpoint=='sensitivity') a<-a[a$reference_positive==1,]
   if (endpoint=='specificity') a<-a[a$reference_positive==0,]
   a$outcome <- as.integer(a$correct_overlap)
   if (length(unique(a$outcome))<2) stop('Constant binary outcome: logistic regression is not estimable.')
   a$strategy<-factor(a$strategy,levels=levels)
   a$seed<-factor(a$seed);a$group_id<-factor(a$group_id);a$sample_id<-factor(a$sample_id);a$model_id<-factor(a$model_id)
   if (any(table(a$strategy)==0)) stop('All nine evaluated groups are required.')
   paired<-lapply(levels,function(label)sort(paste(a$sample_id[a$strategy==label],a$seed[a$strategy==label],a$fold[a$strategy==label],sep='|')))
   if(!all(vapply(paired,function(x)identical(x,paired[[1]]),logical(1)))) stop('All strategies require identical paired case/seed/fold evaluations.')
   fixed <- if (nlevels(a$seed)>1) 'strategy + seed' else 'strategy'
   nuisance <- if (nlevels(a$seed)>1) 'seed' else '1'
   a$bitewing_id <- factor(a$bitewing_id)
   random <- '(1|bitewing_id) + (1|sample_id) + (1|model_id)'
   if ('patient_id' %in% names(a) && all(!is.na(a$patient_id) & nzchar(a$patient_id))) {
    mapping <- unique(a[c('patient_id','bitewing_id')])
    if (any(table(mapping$patient_id)>1)) {a$patient_id<-factor(a$patient_id);random<-paste(random,'+ (1|patient_id)')}
   }
   full_formula<-as.formula(paste('outcome ~',fixed,'+',random))
   null_formula<-as.formula(paste('outcome ~',nuisance,'+',random))
   warnings<-character(0)
   fit<-withCallingHandlers(lme4::glmer(full_formula,data=a,family=binomial(),nAGQ=1,
      control=lme4::glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000))),
      warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
   null<-lme4::glmer(null_formula,data=a,family=binomial(),nAGQ=1,
      control=lme4::glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000)))
   singular<-lme4::isSingular(fit)||lme4::isSingular(null)
   converged<-is.null(fit@optinfo$conv$lme4$messages)&&is.null(null@optinfo$conv$lme4$messages)
   extreme<-any(abs(lme4::fixef(fit))>20)||any(sqrt(diag(vcov(fit)))>10)
   valid<-converged&&!singular&&!extreme
   emm<-emmeans::emmeans(fit,~strategy,weights='equal')
   ct<-as.data.frame(summary(emmeans::contrast(emm,method=contrasts),infer=c(TRUE,TRUE),adjust='none'))
   ct$p_holm<-p.adjust(ct$p.value,method='holm');ct$odds_ratio<-exp(ct$estimate)
   ct$comparison_family_size<-length(contrasts)
   ct$planned_kd_contrast<-ct$contrast %in% c('D - D0','C - C0','F - E')
   ct$interpretation<-ifelse(ct$planned_kd_contrast,'planned KD contrast: matched starting weights/data/training budget',
       ifelse(ct$contrast %in% c('A - D','B - C'),'KD plus additional clinical training; does not isolate KD','complete strategy comparison'))
   lo<-if('asymp.LCL'%in%names(ct)) 'asymp.LCL' else 'lower.CL'
   hi<-if('asymp.UCL'%in%names(ct)) 'asymp.UCL' else 'upper.CL'
   ct$odds_ratio_ci_low<-exp(ct[[lo]]);ct$odds_ratio_ci_high<-exp(ct[[hi]])
   ct$diagnostics_passed<-valid
   if(!valid){ct$p_holm<-NA;ct$p.value<-NA}
   write.csv(ct,file.path(dest,'planned_contrasts.csv'),row.names=FALSE)
   lrt<-as.data.frame(anova(null,fit,test='Chisq'));lrt$diagnostics_passed<-valid
   if(!valid && 'Pr(>Chisq)' %in% names(lrt)) lrt[['Pr(>Chisq)']]<-NA_real_
   write.csv(lrt,file.path(dest,'global_lrt.csv'))
   write.csv(as.data.frame(lme4::VarCorr(fit)),file.path(dest,'random_effects.csv'),row.names=FALSE)
   capture.output(summary(fit),file=file.path(dest,'model_summary.txt'))
   jsonlite::write_json(list(status=if(valid)'DIAGNOSTICS_PASSED' else 'REQUIRES_STATISTICAL_REVIEW',
       converged=converged,singular=singular,extreme_coefficients=extreme,warnings=warnings,
       formula=paste(deparse(full_formula),collapse=' '),binary_endpoint=endpoint,
       pairwise_comparisons=length(contrasts),planned_kd_contrasts=c('D - D0','C - C0','F - E'),
       multiplicity=paste('Holm correction for',length(contrasts),'pairwise contrasts within endpoint/threshold/cohort'),
       note='No observer effect: these are model predictions. Group is patient when available, otherwise bitewing. GLMM estimates are conditional and asymptotic.'),
       file.path(dest,'diagnostics.json'),pretty=TRUE,auto_unbox=TRUE)
  },error=function(e){jsonlite::write_json(list(status='NOT_ESTIMABLE',error=conditionMessage(e)),file.path(dest,'diagnostics.json'),pretty=TRUE,auto_unbox=TRUE)})
 }
}
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))
