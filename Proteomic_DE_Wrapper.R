generate_DEPs <- function(sample_fname)
{
  print(sample_fname)
  # We will have to re-load this workspace everytime, because factors and eset need to be set to correct value.
  load("C:/.../PB_3Way_Unmatched_Ensembl.rdata")
  res1 = generate_DEPs_PC(Eset, factors, sample_fname, "AH", "CT")
  #print(colnames(res1))
  # We will have to re-load this workspace everytime, because factors and eset need to be set to correct value.
  load("C:/.../PB_3Way_Unmatched_Ensembl.rdata")
  res2 = generate_DEPs_PC(Eset, factors, sample_fname, "AH", "AC")
  #print(colnames(res2))
  # We will have to re-load this workspace everytime, because factors and eset need to be set to correct value.
  load("C:/.../PB_3Way_Unmatched_Ensembl.rdata")
  res3 = generate_DEPs_PC(Eset, factors, sample_fname, "AC", "CT")
  
  #print(colnames(res1))
  #print(colnames(res2))
  #print(colnames(res3))
  
  result = rbind(res1, res2, res3)
  
  fname_out = paste("C:/.../", sample_fname, "_DE.csv", sep = "")
  write.csv(result, fname_out, row.names=TRUE)
}

generate_DEPs_ALL <- function()
{
  #a = c("Sample_Train1", "Sample_Train2", "Sample_Train3", "Sample_Train4", "Sample_Train5")
  #a1 = c("Sample_Train1_1", "Sample_Train1_2", "Sample_Train1_3", "Sample_Train1_4", "Sample_Train1_5")
  #a2 = c("Sample_Train2_1", "Sample_Train2_2", "Sample_Train2_3", "Sample_Train2_4", "Sample_Train2_5")
  #a3 = c("Sample_Train3_1", "Sample_Train3_2", "Sample_Train3_3", "Sample_Train3_4", "Sample_Train3_5")
  #a4 = c("Sample_Train4_1", "Sample_Train4_2", "Sample_Train4_3", "Sample_Train4_4", "Sample_Train4_5")
  #a5 = c("Sample_Train5_1", "Sample_Train5_2", "Sample_Train5_3", "Sample_Train5_4", "Sample_Train5_5")
  a = c("Sample_Train1", "Sample_Train2", "Sample_Train3")
  a1 = c("Sample_Train1_1", "Sample_Train1_2", "Sample_Train1_3")
  a2 = c("Sample_Train2_1", "Sample_Train2_2", "Sample_Train2_3")
  a3 = c("Sample_Train3_1", "Sample_Train3_2", "Sample_Train3_3")
  
  a = append(a, a1)
  a = append(a, a2)
  a = append(a, a3)
  #a = append(a, a4)
  #a = append(a, a5)
  
  cnt = 1 
  while(cnt <= length(a))
  {
    generate_DEPs(a[cnt])
    cnt = cnt + 1
  }
  
}