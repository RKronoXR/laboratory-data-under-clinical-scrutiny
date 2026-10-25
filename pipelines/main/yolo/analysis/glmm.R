#!/usr/bin/env Rscript
# Binary localization correctness, sensitivity and specificity only.
# AUROC/AP/Dice/MCC contrasts are evaluated by the paired cluster bootstrap.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) stop('Usage: Rscript analysis/glmm.R surface_results.csv output_dir')
needed <- c('lme4','emmeans','jsonlite')
if (any(!vapply(needed,requireNamespace,logical(1),quietly=TRUE))) stop('Missing R packages. Run INSTALL_STATISTICS.sh')
out <- args[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
d <- read.csv(args[1],stringsAsFactors=FALSE)
contrasts <- list('B - A'=c(0,-1,1,0,0),'D - C'=c(0,0,0,-1,1),'C - A'=c(0,-1,0,1,0),
  'D - B'=c(0,0,-1,0,1),'A - T'=c(-1,1,0,0,0),'B - T'=c(-1,0,1,0,0),
  'C - T'=c(-1,0,0,1,0),'D - T'=c(-1,0,0,0,1))
for (setting in unique(d$threshold_name)) {
 for (endpoint in c('correctness','sensitivity','specificity')) {
  dest <- file.path(out,paste(setting,endpoint,sep='_'));dir.create(dest,recursive=TRUE,showWarnings=FALSE)
  tryCatch({
   a <- d[d$threshold_name==setting,]
   if (endpoint=='sensitivity') a<-a[a$reference_positive==1,]
   if (endpoint=='specificity') a<-a[a$reference_positive==0,]
   a$outcome <- as.integer(a$correct_overlap)
   if (length(unique(a$outcome))<2) stop('Constant binary outcome: logistic regression is not estimable.')
   a$strategy<-factor(a$strategy,levels=c('T','A','B','C','D'))
   a$seed<-factor(a$seed);a$group_id<-factor(a$group_id);a$sample_id<-factor(a$sample_id);a$model_id<-factor(a$model_id)
   if (any(table(a$strategy)==0)) stop('All five evaluated groups are required.')
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
       multiplicity='Holm correction for eight planned contrasts within endpoint/threshold/cohort',
       note='No observer effect: these are model predictions. Group is patient when available, otherwise bitewing. GLMM estimates are conditional and asymptotic.'),
       file.path(dest,'diagnostics.json'),pretty=TRUE,auto_unbox=TRUE)
  },error=function(e){jsonlite::write_json(list(status='NOT_ESTIMABLE',error=conditionMessage(e)),file.path(dest,'diagnostics.json'),pretty=TRUE,auto_unbox=TRUE)})
 }
}
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))
