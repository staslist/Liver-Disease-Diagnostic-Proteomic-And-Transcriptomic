load("C:/.../Inferno.RData")
load("C:/.../Inferno_stdplots.RData")

# We will have to re-load this workspace everytime, because factors and eset need to be set to correct value.
# load("C:/.../PB_3Way_Unmatched_Ensembl.rdata")

select_factors <- function(factors, cond_label){
  #print("In select factors!")
  #print(factors)
  #print(cond_label)
  cnt = 1
  a = c()
  while(cnt <= length(factors))
  {
    if(factors[1,cnt]==cond_label)
    {
      a = append(a, cnt)
    }
    cnt = cnt + 1
  }
  #print(a)
  #print("Select factors over!")
  return(a)
}

generate_DEPs_PC <- function(Eset, factors, sample_fname, cond1_label, cond2_label)
{
  fname = paste("C:/.../", sample_fname, ".csv", sep = "")
  sample_names <- read.csv(fname)
  
  #print(sample_names)
  
  # Generate sub-sets of Eset and factors using sample_names read in from the file. 
  # Perform fold changes, p-value, and q-values on these subsets. 
  # Form DE files.
  a = c(colnames(sample_names)[1])
  
  cnt <- 1
  limit <- nrow(sample_names)
  
  while (cnt <= limit) {
    a = append(a, sample_names[cnt,,][1])
    cnt = cnt + 1
  }
  a = unlist(a)
  
  #print(a)
  #print(factors)
  
  Eset <<- Eset[,a]
  factors <<- factors[,a,drop=FALSE]
  Eset = Eset[,a]
  factors = factors[,a,drop=FALSE]
  
  #print(factors)
  
  b = select_factors(factors, cond1_label)
  b = append(b, select_factors(factors, cond2_label))
  
  #print(dim(Eset))
  #print(dim(factors))
  
  Eset <<- Eset[,b]
  factors <<- factors[,b,drop=FALSE]
  Eset = Eset[,b]
  factors = factors[,b,drop=FALSE]
  
  #print(b)
  #print(factors)
  
  #print(dim(Eset))
  #print(dim(factors))
  
  anova_results = DoAnova(Eset, "Conditions", NULL, thres = 1)
  FC = calcFoldChanges(Eset, factors, cond1_label, cond2_label, FALSE)
  FC_sub = FC[,1:5]
  result = merge(FC_sub, anova_results$pvals, by = 'row.names', all = TRUE)
  rownames(result) = result[,1]
  result = result[,2:8]
  result[,8] = rep(c(cond1_label), nrow(result))
  colnames(result)[which(names(result) == "V8")] <- "cond1_label"
  result[,9] = rep(c(cond2_label), nrow(result))
  colnames(result)[which(names(result) == "V9")] <- "cond2_label"
  result[,10] = rep(c(table(factors)[cond1_label]), nrow(result))
  colnames(result)[which(names(result) == "V10")] <- "cond1_samples"
  result[,11] = rep(c(table(factors)[cond2_label]), nrow(result))
  colnames(result)[which(names(result) == "V11")] <- "cond2_samples"
  
  colnames(result)[which(names(result) == cond1_label)] <- "count1"
  colnames(result)[which(names(result) == cond2_label)] <- "count2"
  
  colnames(result)[which(names(result) == paste(cond1_label, "(#non-missing)", sep = ""))] <- "cond1(#non-missing)"
  colnames(result)[which(names(result) == paste(cond2_label, "(#non-missing)", sep = ""))] <- "cond2(#non-missing)"
  
  return(result)
}