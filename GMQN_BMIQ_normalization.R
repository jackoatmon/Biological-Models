library(minfi)
library(gmqn)
library(stringr)
library(IlluminaHumanMethylationEPICmanifest)
library(impute)

patient = c("205676380148_R08C01")
fileID = str_split(patient,"_")[[1]][1]

# working directory 
setwd('C:/Users/jack/PycharmProjects/TruDiagnostic/AltumAge')

pat_ages = read.csv("PopulationData_050722.csv",header = TRUE)[,c(1:5,9)]
pat_ages = pat_ages[order(pat_ages$Patient.ID),]
pat_ages = pat_ages[grep(fileID, pat_ages$Patient.ID),]

file_loc_rg = paste0("C:/Users/jack/PycharmProjects/TruDiagnostic/AltumAge/Data/",fileID,"/",pat_ages$Patient.ID,"_Grn.idat")
rgSet = read.metharray(basenames = file_loc_rg, extended = TRUE)
raw_betas = getBeta(rgSet)
MSet = preprocessRaw(rgSet) 
BMIQ_betas = bmiq_parallel(data.frame(getMeth(MSet)), data.frame(getUnmeth(MSet)), type = "850k", ref = "default", ncpu = 4)
BMIQ_betas_imp = data.frame(impute.knn(as.matrix(BMIQ_betas))$data)
colnames(BMIQ_betas_imp) = sub("X","",colnames(BMIQ_betas_imp))


write.csv(BMIQ_betas_imp, 'PopDataTest.csv')





 
