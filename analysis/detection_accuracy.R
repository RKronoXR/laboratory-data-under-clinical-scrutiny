# Exploratory paired multilevel logistic analysis of regional detection accuracy.
# Rscript analysis/detection_accuracy.R [regional_predictions.csv.gz] [output_directory] [--no-reference-check]
# No images, trained weights or GPU are needed for this analysis.
custom_lib <- Sys.getenv('CARIES_R_LIB', '')
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
stopifnot(requireNamespace('lme4', quietly=TRUE), requireNamespace('jsonlite', quietly=TRUE))
options(stringsAsFactors=FALSE, contrasts=c('contr.treatment','contr.poly'))
argfile <- grep('^--file=', commandArgs(), value=TRUE)
repo <- if(length(argfile)) dirname(dirname(normalizePath(sub('^--file=', '', argfile[1])))) else getwd()
args <- commandArgs(trailingOnly=TRUE)
reference_check <- !('--no-reference-check' %in% args)
args <- args[args!='--no-reference-check']
if(any(grepl('^--',args)) || length(args)>2)stop('Usage: Rscript analysis/detection_accuracy.R [input.csv.gz] [output_directory] [--no-reference-check]')
input <- if(length(args)>=1) args[1] else file.path(repo,'data','reference','main_region_predictions.csv.gz')
outdir <- if(length(args)>=2) args[2] else file.path(repo,'results','statistics')
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
stopifnot(file.exists(input))
con <- if(grepl('\\.gz$',input)) gzfile(input,'rt') else file(input,'rt')
all <- read.csv(con,check.names=FALSE);close(con)
required <- c('architecture','strategy','seed','fold','region','bitewing','threshold_mode','threshold','reference_positive','reference_pixels','predicted_pixels','intersection_pixels','correct_overlap')
stopifnot(all(required %in% names(all)))
dat <- all[all$threshold_mode=='fixed',,drop=FALSE]
rm(all)
stopifnot(nrow(dat)==44280L,all(dat$threshold==0.5),all(dat$reference_positive %in% 0:1),all(dat$correct_overlap %in% 0:1))
expected <- ifelse(dat$reference_positive==1,dat$intersection_pixels>0,dat$predicted_pixels==0)
stopifnot(all(dat$correct_overlap==as.integer(expected)))
stopifnot(all(dat$intersection_pixels>=0),all(dat$intersection_pixels<=dat$predicted_pixels),all(dat$intersection_pixels<=dat$reference_pixels))
public_keys <- c(Clinical='clinical_only',Pretraining='laboratory_pretraining',Distillation='knowledge_distillation',Combined='pretraining_and_distillation')
canonical <- setNames(names(public_keys),public_keys)
stopifnot(all(dat$strategy %in% names(canonical)),all(dat$architecture %in% c('nnU-Net','YOLO')))
dat$strategy <- factor(unname(canonical[dat$strategy]),levels=names(public_keys))
dat$run <- paste0('seed',dat$seed,'_fold',dat$fold)
dat$fit_id <- paste(dat$architecture,dat$strategy,dat$run,sep=':')
dat$region_strategy <- paste(dat$strategy,dat$region,sep=':')
stopifnot(!anyDuplicated(paste(dat$fit_id,dat$region,sep=':')))
stopifnot(length(unique(dat$region))==369L,length(unique(dat$bitewing))==25L,length(unique(dat$fit_id))==120L)
stopifnot(all(table(dat$architecture,dat$strategy)==15L*369L),all(table(dat$region,dat$architecture,dat$strategy)==15L))
stopifnot(setequal(unique(dat$seed),c(17, 42, 2026)),setequal(unique(dat$fold),0:4))
region_reference <- unique(dat[,c('region','bitewing','reference_positive','reference_pixels')])
stopifnot(nrow(region_reference)==369L,sum(region_reference$reference_positive)==184L)
for(v in c('run','fit_id','region','bitewing','region_strategy'))dat[[v]]<-factor(dat[[v]])
write.csv(dat,file.path(outdir,'data_used.csv'),row.names=FALSE)
# Compute each model's metric, average five folds within seed, then summarise seeds.
spec <- data.frame(architecture=rep(c('nnU-Net','YOLO'),each=4),strategy=rep(names(public_keys),2))
descriptive <- list()
for(i in seq_len(nrow(spec))) {
 x <- dat[dat$architecture==spec$architecture[i] & dat$strategy==spec$strategy[i],]
 by_run <- aggregate(correct_overlap~seed+fold,data=x,FUN=mean)
 by_seed <- aggregate(correct_overlap~seed,data=by_run,FUN=mean)
 descriptive[[i]] <- data.frame(architecture=spec$architecture[i],strategy=spec$strategy[i],outcome='accuracy',mean=mean(by_seed$correct_overlap),sd=sd(by_seed$correct_overlap))
}
desc <- do.call(rbind,descriptive)
desc_public <- desc;desc_public$strategy <- unname(public_keys[desc_public$strategy])
write.csv(desc_public,file.path(outdir,'descriptive_verified.csv'),row.names=FALSE)
# Three planned contrasts per architecture against clinical-only training.
# Fixed threshold 0.5; no threshold recalibration. Nominal two-sided P values and
# pointwise 95% Wald intervals. Starting random intercepts account for bitewing,
# region, region-by-strategy, matched seed-fold run and fitted model. Only
# components estimated below 1e-8 are omitted, without inspecting effect direction.
fit_warnings<-character()
capture_fit <- function(formula,x,optimizer='bobyqa',start=NULL) {
 localwarnings<-character()
 fit<-withCallingHandlers(lme4::glmer(formula,data=x,family=binomial(link='logit'),nAGQ=1L,
  control=lme4::glmerControl(optimizer=optimizer,optCtrl=list(maxfun=200000L),calc.derivs=TRUE),start=start),
  warning=function(w){localwarnings<<-c(localwarnings,conditionMessage(w));invokeRestart('muffleWarning')})
 list(fit=fit,warnings=localwarnings)
}
all_results<-list();diagnostics<-list();counter<-0L
for(arch in c('nnU-Net','YOLO')) {
 outcome<-'accuracy'
 x<-droplevels(dat[dat$architecture==arch,])
 groups<-c('bitewing','region','region_strategy','run','fit_id');removed<-character();history<-list()
 cat('\nFitting',arch,outcome,'n=',nrow(x),'\n');flush.console()
 repeat {
  formula<-as.formula(paste('correct_overlap ~ strategy +',paste(sprintf('(1|%s)',groups),collapse=' + ')))
  attempt<-capture_fit(formula,x)
  model<-attempt$fit
  vc<-as.data.frame(lme4::VarCorr(model))
  zero<-vc$grp[is.na(vc$var2)&vc$vcov<1e-8]
  history[[length(history)+1L]]<-list(formula=paste(deparse(formula),collapse=' '),warnings=attempt$warnings,variance_components=vc,removed_as_zero=zero)
  if(length(zero)) {
   cat('Omitting zero-variance component(s):',paste(zero,collapse=', '),'\n');flush.console()
   removed<-c(removed,zero);groups<-setdiff(groups,zero);next
  }
  break
 }
 conv<-model@optinfo$conv$lme4$messages
 if(length(conv)) {
  cat('Checking same model with alternate optimizer\n');flush.console()
  alternate<-capture_fit(formula,x,optimizer='Nelder_Mead',start=list(theta=lme4::getME(model,'theta'),fixef=lme4::fixef(model)))
  history[[length(history)+1L]]<-list(optimizer='Nelder_Mead',warnings=alternate$warnings,max_fixed_effect_difference=max(abs(lme4::fixef(alternate$fit)-lme4::fixef(model))),logLik_difference=as.numeric(logLik(alternate$fit)-logLik(model)))
  if(!length(alternate$fit@optinfo$conv$lme4$messages) && as.numeric(logLik(alternate$fit))>=as.numeric(logLik(model))-1e-5)model<-alternate$fit
 }
 mat<-coef(summary(model));ok<-!length(model@optinfo$conv$lme4$messages) && !lme4::isSingular(model,tol=1e-4)
 eig<-eigen(model@optinfo$derivs$Hessian,symmetric=TRUE,only.values=TRUE)$values
 ok<-ok&&all(is.finite(eig))&&min(eig)>0
 key<-paste(gsub('-','',arch),outcome,sep='_')
 saveRDS(model,file.path(outdir,paste0(key,'.rds')))
 diagnostics[[key]]<-list(architecture=arch,outcome=outcome,n_records=nrow(x),n_regions=length(unique(x$region)),n_bitewings=length(unique(x$bitewing)),n_fitted_models=length(unique(x$fit_id)),formula=paste(deparse(formula),collapse=' '),zero_components_removed=removed,variance_components=as.data.frame(lme4::VarCorr(model)),history=history,convergence_messages=model@optinfo$conv$lme4$messages,singular=lme4::isSingular(model),minimum_hessian_eigenvalue=min(eig),valid=ok,logLik=as.numeric(logLik(model)))
 for(strategy in c('Pretraining','Distillation','Combined')) {
  row<-paste0('strategy',strategy);beta<-mat[row,'Estimate'];se<-mat[row,'Std. Error'];z<-beta/se
  base<-desc$mean[desc$architecture==arch & desc$strategy=='Clinical' & desc$outcome==outcome]
  current<-desc$mean[desc$architecture==arch & desc$strategy==strategy & desc$outcome==outcome]
  counter<-counter+1L
  all_results[[counter]]<-data.frame(architecture=arch,outcome=outcome,strategy=strategy,baseline_mean=base,strategy_mean=current,difference_percentage_points=100*(current-base),log_odds_ratio=beta,se=se,odds_ratio=exp(beta),or_lower=exp(beta-qnorm(.975)*se),or_upper=exp(beta+qnorm(.975)*se),z=z,p_nominal=2*pnorm(-abs(z)),model_valid=ok)
 }
 cat('Completed',arch,outcome,'valid=',ok,'\n');flush.console()
 jsonlite::write_json(diagnostics,file.path(outdir,'diagnostics.json'),pretty=TRUE,auto_unbox=TRUE,na='null',digits=NA)
}
results<-do.call(rbind,all_results)
stopifnot(nrow(results)==6L,all(results$outcome=='accuracy'))
results$strategy <- unname(public_keys[results$strategy])
write.csv(results,file.path(outdir,'contrasts.csv'),row.names=FALSE)

versions <- list(R=R.version.string,lme4=as.character(packageVersion('lme4')),jsonlite=as.character(packageVersion('jsonlite')))
audit <- list(input_file=basename(input),input_md5=unname(tools::md5sum(input)),n_records=nrow(dat),n_regions=369,n_positive=184,n_negative=185,n_bitewings=25,n_models=120,threshold=0.5,comparison_scope='Six planned contrasts of regional detection accuracy against architecture-specific clinical-only baselines',multiplicity='Nominal exploratory P values and pointwise 95% intervals; no multiplicity adjustment',reference_comparison_requested=reference_check,versions=versions)
jsonlite::write_json(list(audit=audit,descriptive=desc_public,results=results,diagnostics=diagnostics),file.path(outdir,'analysis_results.json'),pretty=TRUE,auto_unbox=TRUE,na='null',digits=NA)
capture.output(sessionInfo(),file=file.path(outdir,'sessionInfo.txt'))
expected_path <- file.path(repo,'data','reference','expected_published_summary.csv')
if(reference_check) {
 if(!file.exists(expected_path))stop('The published reference summary is missing. Restore data/reference/expected_published_summary.csv, or explicitly use --no-reference-check for new model results.')
 reference <- read.csv(expected_path)
 check <- merge(results,reference,by=c('architecture','strategy','outcome'),suffixes=c('_actual','_expected'))
 stopifnot(nrow(check)==6L)
 # Tolerance accommodates BLAS/optimizer variation while preserving publication rounding.
 for(v in c('odds_ratio','or_lower','or_upper','p_nominal'))stopifnot(max(abs(check[[paste0(v,'_actual')]]-check[[paste0(v,'_expected')]]))<1e-4)
 write.csv(check,file.path(outdir,'published_result_comparison.csv'),row.names=FALSE)
} else {
 stale_comparison <- file.path(outdir,'published_result_comparison.csv')
 if(file.exists(stale_comparison))unlink(stale_comparison)
 cat('Published-value comparison disabled explicitly for new model results.\n')
}
cat('\nAll six detection-accuracy contrasts saved. All models valid:',all(results$model_valid),'\n')
if(!all(results$model_valid))stop('At least one fitted model failed convergence or singularity checks.')
