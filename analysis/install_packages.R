# Install the recorded analysis package versions; R 4.5.1 was used in the study.
# Linux source installs need the usual R compilation toolchain. On Windows use
# matching CRAN binaries where available, or install Rtools for source builds.
argfile <- grep('^--file=',commandArgs(),value=TRUE)
here <- if(length(argfile)) dirname(normalizePath(sub('^--file=','',argfile[1]))) else 'analysis'
versions <- read.csv(file.path(here,'package_versions.csv'),stringsAsFactors=FALSE)
repos <- 'https://cloud.r-project.org'
if(!requireNamespace('remotes',quietly=TRUE))install.packages('remotes',repos=repos)
for(i in seq_len(nrow(versions))) {
 name <- versions$package[i];version <- versions$version[i]
 matching <- requireNamespace(name,quietly=TRUE) && packageVersion(name)==package_version(version)
 if(!matching)remotes::install_version(name,version=version,repos=repos,upgrade='never',dependencies=NA)
}
cat('Recorded analysis package versions are installed.\n')
