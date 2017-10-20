## Generate Heatmap for Pairwise Correlation
## To Run: R --no-save < heatmap_cluster.R --args data_PCA.txt PCA_Pearson pearson
## Input: data_PCA.txt - input data
## 	  PCA_Pearson - output folder
##        pearson - correlation method
##



library(RColorBrewer)
library(lattice)
library(reshape2)
library(plyr)


options <- commandArgs(trailingOnly = TRUE);
if(length(options) != 3){
        stop(paste("Invalid Arguments\n",
        "Usage: R --no-save --slave < heatmap_cluster.R  --args input outdir method\n",
	"\t input = input filename\n",
	"\t outdir = output directory\n",
	sep=""));
}

infile <- options[1];
outdir <- options[2];
method <- options[3];

data <- read.table(infile);
data_correlation <- cor(t(data), method=method);
rownames(data_correlation) <- seq(1,nrow(data_correlation),1)

fit <- hclust(dist(data_correlation))
pairwise_reorder <- data_correlation[fit$order, fit$order]

## cut the trees
branches <- 12
subcluster <- cutree(fit, k = branches)
subcluster_label <- fit$label[fit$order]

## plot heatmap
rgb.palette <- colorRampPalette(brewer.pal(11, "RdBu"))
pdf(paste(outdir, "heatmap.pdf", sep="/"))
levelplot(pairwise_reorder, at = unique(c(seq(-1.1, 1.1, length = 100))), col.regions=rgb.palette(100), cuts=50, 
          scales=list(x=list(cex=.1, rot=60), y=list(cex=.1), tck=c(0,0)),
          xlab = "Locations", ylab="Locations", main = "Pairwise Correlations")
dev.off()

## plot dendrogram
pdf(paste(outdir, "dendrogram.pdf", sep="/"))
plot(fit, cex = 0.2, main = "Cluster of Pairwise Correlation" )
rect.hclust(fit, k=branches, border = "red")
dev.off()


for(x in 1:branches){
	subtree <- subcluster_label[subcluster_label %in% rownames(as.matrix(subcluster[subcluster == x]))]
	subtree_data <- data[subtree,]

	pdf(paste(outdir, "/heatmap_subcluster_", x, ".pdf", sep="")) 
	print(
	levelplot( cor(t(subtree_data)), at = unique(c(seq(-1.1, 1.1, length = 100))), col.regions=rgb.palette(100), cuts=50,
          scales=list(x=list(cex=.5, rot=60), y=list(cex=.5), tck=c(0,0)),
          xlab = "Locations", ylab="Locations", main = "Pairwise Correlations") 
	)
	dev.off()
}

write.table(subcluster, paste(outdir, "/cluster_label.txt", sep=""), sep="\t")

