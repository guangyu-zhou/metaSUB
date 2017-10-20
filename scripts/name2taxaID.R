library(myTAI)
setwd('/Users/DavidZhou/GDriveUCLA/Study/Research/MetaGenomic/MetaSUB/ny/scripts/')
#species = read.table("../files/species_names_tree.txt", sep="\n")

retrieve <- function(){
  species = read.table("../files/missing_ID.txt", sep="\n")
  for (i in 1:length(species$V1)) { 
    #print(species$V1[i])
    a = taxonomy( organism = species$V1[i], 
                  db       = "ncbi",
                  output   = "classification" )
    #print(a$rank)
    #print(a[a$rank == 'genus'],)
    print(i)
    #print(row.names(a).length)
    #print(subset(a, rank == "species"))
    print(a)
  }
}


#species$V1[2]
#length(species$V1)
retrieve2 <- function(){
  species = read.table("../files/not_found_classification.txt", sep="\n")
  for (i in 1:length(species$V1)) { 
    #print(species$V1[i])
    a = taxonomy( organism = species$V1[i], 
                  db       = "itis",
                  output   = "classification" )
    #print(a$rank)
    #print(a[a$rank == 'genus'],)
    print(i)
    #print(row.names(a).length)
    #print(subset(a, rank == "species"))
    print(a)
  }
}
#retrieve()
retrieve2()




