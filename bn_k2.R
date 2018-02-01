#!/usr/bin/env Rscript
args = commandArgs(TRUE)
#options(echo=FALSE)
library('bnlearn')

trim <- function (x) gsub("^\\s+|\\s+$", "", x)

remove.dots <- function (x) gsub("\\.", "", x)

process.row <- function(row){
  fname <- row[1]
  ftype <- remove.dots(trim(row[2]))
  if (length(row) == 3)
  {
    fdomain <- split(remove.dots(row[3]), ',')
    print(class(fdomain))
  }
  else
  {
    fdomain <- ''
  }
  row[1] <- fname
  row[2] <- ftype
  row[3] <- fdomain
  row
}

collect.names <- function(row){
  trim(row[1])
}

collect.families <- function(row){
  trim(remove.dots(row[2]))
}

collect.domains <- function(row){
  if (length(row) == 3)
  {
    trim(substr(row[3], 1, nchar(row[3])-1))
  }
  else
  {
    ''
  }
}

load.feature.info <- function(data.path){
  df <- read.delim(data.path, sep=':',col.names=c("feature.name","feature.type","feature.domain"), header=F, strip.white=T)
  names <- apply(df, 1, collect.names)
  families <- apply(df, 1, collect.families)
  domains <- apply(df, 1, collect.domains)
  domains <- lapply(domains, function(x) strsplit(x, ',')[[1]])
  list(names, families, domains)
}

data.transformer <- function(data.frame, feature.names, feature.families, feature.domains){
  for (i in 1:length(feature.names))
  {
    if (feature.families[i] == 'continuous')
    {
      data.frame[,feature.names[i]] <- as.numeric(data.frame[,feature.names[i]])
    }
    else if (feature.families[i] == 'discrete')
    {
      data.frame[,feature.names[i]] <- factor(data.frame[,feature.names[i]], levels=feature.domains[[i]], ordered=T)
    }
    else if (feature.families[i] == 'categorical')
    {
      data.frame[,feature.names[i]] <- factor(data.frame[,feature.names[i]], levels=feature.domains[[i]], ordered=F)
    }
  }
  data.frame
}

is.discrete <-function(type){
  return(type == 'discrete')
}

get.discrete.data <- function(feature.names, feature.families){
  feature.names[lapply(feature.families, is.discrete) == TRUE]
}

load.dataset <- function(data.path, featname){
    info <- load.feature.info(featname)
    data <- read.table(paste(data.path, '.data', sep=''), sep=",", header=F, col.names=info[[1]], fill=FALSE, strip.white=T, check.names=FALSE)
    data <- data.transformer(data, info[[1]], info[[2]], info[[3]])
    return (data)
}

featuresname <- args[1]
traindataname <- args[2]
testdataname <- args[3]
outfile <- args[4]


eval.bn <- function(bn, data){
  n.instances <- dim(data)[1]
  ll <- logLik(bn, data)
  avg.ll <- ll / n.instances
  return(avg.ll)
}

ll.bn <- function(bn, data){
	ll <- logLik(bn, data, by.sample=TRUE)
	return(ll)
}

learn.bn <- function(D, learn.method='mmhc'){
  ## learning the network structure
  if (learn.method == 'mmhc')
  {
    net <- mmhc(D)
  }
  else if (learn.method == 'hc')
  {
    net <- hc(D, score='k2')
  }
  else if (learn.method == 'tabu')
  {
    net <- tabu(D)
  }
  else if (learn.method == 'gs')
  {
    net <- gs(D)
  }
  else if (learn.method == 'iamb')
  {
    net <- iamb(D)
  }
  else if (learn.method == 'mmpc')
  {
    net <- mmpc(D)
  }
  else if (learn.method == 'si.hiton.pc')
  {
    net <- si.hiton.pc(D)
  }
  else if (learn.method == 'rsmax2')
  {
    net <- rsmax2(D)
  }
 
  ## learning the network weights
  net.fit <- bn.fit(net, D, method='bayes', iss=1)
  return(net.fit)
}

Train <- load.dataset(traindataname, featuresname)
Test <- load.dataset(testdataname, featuresname)

Total <- rbind(Train, Test)

Discretized = Total

train_rows = nrow(Train)
D = Discretized[1:train_rows,]
D1 = Discretized[(train_rows+1):nrow(Total),]

net <- learn.bn(D, learn.method='hc')

lls <- ll.bn(net, D1)
write.table(lls, outfile, row.names=FALSE, col.names=FALSE)

