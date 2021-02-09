library(tidyverse)

# assumes you have a data frame "oj.train" with numeric predictors


oj.missing <- oj.train %>% as.matrix() # must be a matrix for this to work
sum(oj.missing==0) # how many zeros are there already?
dim(oj.missing) 
length(oj.missing) # matrixes are really vectors that flow into a 2D space

# put 0 in 10% of the cells
oj.missing[sample(length(oj.missing), size = 0.1*length(oj.missing))] <- 0

dim(oj.missing)
sum(oj.missing==0)