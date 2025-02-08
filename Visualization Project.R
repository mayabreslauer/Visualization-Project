# Load required packages
library(DBI)
# install.packages("DBI")
# install.packages("statnet")
library(dplyr)
library(sqldf)
library(RMySQL)
library(stringi)
library(sna)
library(network)
library(igraph)
library(intergraph)
library(statnet)
library(ergm)
install.packages("writexl")


# Connect to the database
con <- dbConnect(MySQL(), 
                 user = "root", 
                 password = "mayaB1406!", 
                 dbname = "visualization", 
                 host = "localhost",)

table<-dbGetQuery(con,"select * from ages as a join eshkol as e on a.שם_ישוב=e.City_Name join religion as r on r.שם_יישוב =e.City_Name")
# Sys.setlocale(category = "LC_ALL", locale = "Hebrew_Hebrew.65001")
library(writexl)
write_xlsx(table, "output_file1.xlsx")


