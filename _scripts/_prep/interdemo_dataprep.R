# data prep for intervention and democracy (institutions) project
# this is for multinomial model of intervention

library(tidyverse)
library(foreign)
library(readstata13)
library(ggplot2)
library("readxl")
library(countrycode)
library("lubridate")
library(nnet)
library(sandwich)
library(lmtest)
library(stargazer)
library(mlogit)
library(fastDummies)
library(peacesciencer)
library(scales)
library(vdemdata)
rm(list=ls(all=TRUE))

# Creating main dyadic df -------------------------------------------------

# mid and trade data
download_extdata()

# creading dyad-year dataset for the population of potential interveners, starting a year prior to sample for lagging RHS
dyadyear <- create_dyadyears(system = "cow", subset_years = c(1974:2009)) %>%
  add_contiguity() %>%
  add_minimum_distance() %>%
  add_cow_alliance() %>%
  add_cow_majors() %>%
  add_cow_wars(type = "inter") %>%
  select(., -c("sidea1", "sidea2", "initiator1", "initiator2", "outcome1", "outcome2", "batdeath1", "batdeath2", "resume")) %>%
  add_cow_mids(keep = c("cowmidongoing", "cowmidonset")) %>%
  add_cow_trade() %>%
  add_sdp_gdp() %>%
  add_democracy() %>%
  select(., -c("v2x_polyarchy1", "v2x_polyarchy2")) %>%
  add_fpsim(keep = c("kappava", "kappavv")) %>%
  add_strategic_rivalries() %>%
  group_by(ccode1, ccode2) %>%
  mutate(., mindistlead = lead(mindist)) %>% # fixing mindist missing here
  mutate(mindist = ifelse(is.na(mindist), mindistlead, mindist)) %>%
  subset(., select = -c(mindistlead)) %>%
  ungroup()


# RHS variables -----------------------------------------------------------

# gonna pull in all the covariates here, so I can lag them all together
# san-acka, prd, more v-dem, trade openness, rough terrain, regional dummies, ongoing conflicts in ccode2

# san-akca on third-party rebel ties
sanakca <- read.dta13("_data/_raw/BookFinalData.dta")

df.sanakca <- sanakca %>%
  rename(., ccode1 = tarnum, ccode2 = psnum) %>%
  subset(., select = c("ccode1", "ccode2", "year", "PSTID", "PSNAGID")) %>%
  group_by(., ccode1, ccode2, year) %>%
  summarise_all(max, na.rm= TRUE) %>%
  ungroup()
# PSTID is whether T and G have ties, while PSNAGID is whether T and R have ties

# politically relevant dyads
prd <- read.dta13("_data/_raw/prd.dta")
# this stops in 2001

# more vdem
#vdem <- read.csv("/Users/hans-ingelang/Dropbox/Analysis/Data/Vdem/V-Dem-CY-Full+Others-v10.csv")
# don't need this if I'm using the vdem package

# need to check the variables are the same in package
vdem.vars1 <- vdem %>% 
  dplyr::rename(., ccode1 = COWcode, v2x_polyarchy1 = v2x_polyarchy, v2x_liberal1 = v2x_liberal, v2x_libdem1 = v2x_libdem) %>%
  filter(., year>1944) %>%
  select(., ccode1, year, v2x_polyarchy1, v2x_liberal1, v2x_libdem1) %>%
  arrange(., ccode1, year)

vdem.vars2 <- vdem %>% 
  dplyr::rename(., ccode2 = COWcode, v2x_polyarchy2 = v2x_polyarchy, v2x_liberal2 = v2x_liberal, v2x_libdem2 = v2x_libdem) %>%
  filter(., year>1944) %>%
  select(., ccode2, year, v2x_polyarchy2, v2x_liberal2, v2x_libdem2) %>%
  arrange(., ccode2, year)

# merge in trade openness data from Chinn-Ito

open1 <- read.dta13("_data/_raw/open.dta") %>%
  distinct(cowcode, year, .keep_all = TRUE) %>%
  dplyr::rename(., ccode1 = cowcode, kaopen1 = kaopen) %>%
  select(., ccode1, year, kaopen1) %>%
  arrange(., ccode1, year)

open1 <-open1[!is.na(open1$ccode1),]

open1 <-open1[!is.na(open1$year),]

# fix Russia from 365 to 364 because of discrepancy between datasets
open1$ccode1[open1$ccode1==364]=365

open2 <- read.dta13("_data/_raw/open.dta") %>%
  distinct(cowcode, year, .keep_all = TRUE) %>%
  dplyr::rename(., ccode2 = cowcode, kaopen2 = kaopen) %>%
  select(., ccode2, year, kaopen2) %>%
  arrange(., ccode2, year)

open2 <-open2[!is.na(open2$ccode2),]

open2 <-open2[!is.na(open2$year),]

# fix Russia from 365 to 364 because of discrepancy between datasets

open2$ccode2[open2$ccode2==364]=365

# getting regional dummies from Hegre and Sambanis
hg <- read.dta13("_data/_raw/sambanis06.dta") %>%
  distinct(cowcode, year, .keep_all = TRUE) %>%
  dplyr::rename(., ccode1 = cowcode) %>%
  subset(., select = c("ccode1", "geo")) %>%
  group_by(ccode1) %>%
  summarise_all(max, na.rm= TRUE)
# cleaning up
hg$geo[hg$geo==-Inf]=NA
hg <-hg[!is.na(hg$ccode1),]

# add in ongoing wars in ccode2

acdongoingcw <- read.dta13("_data/_raw/acdcwyear.dta") %>%
  rename(., ccode2 = ccode)

acdongoingiw <- read.dta13("_data/_raw/acdiwyear.dta") %>%
  rename(., ccode2 = ccode)

# add in more covariates: i really should clean this all up and consolidate, avoid dupes from peacesciencer
cov1 <- read.dta13("_data/_raw/covariates1.dta") %>%
  select(., c("ccode1", "year", "upop_mc_1", "cinc_mc_1", "milex_mc_1", "milper_mc_1", "growth_wdi_1", "pop_den_wdi_1"))

cov2 <- read.dta13("_data/_raw/covariates2.dta") %>%
  select(., c("ccode2", "year", "upop_mc_2", "cinc_mc_2", "milex_mc_2", "milper_mc_2", "growth_wdi_2", "pop_den_wdi_2"))


# Joining dyad df and RHS -------------------------------------------------

# merge in everything

df.dyadyear <- dyadyear %>% 
  left_join(., prd, by = c("ccode1", "ccode2", "year")) %>%
  left_join(., df.sanakca, by = c("ccode1", "ccode2", "year")) %>%
  left_join(., vdem.vars1, by = c("ccode1", "year")) %>%
  left_join(., vdem.vars2, by = c("ccode2", "year")) %>%
  left_join(., open1, by = c("ccode1", "year")) %>%
  left_join(., open2, by = c("ccode2", "year")) %>%
  left_join(., hg, by = c("ccode1")) %>%
  left_join(., cov1, by = c("ccode1", "year")) %>%
  left_join(., cov2, by = c("ccode2", "year")) %>%
  left_join(., acdongoingcw, by = c("ccode2", "year")) %>%
  mutate(., acdcwyear = ifelse(is.na(acdcwyear), 0, acdcwyear)) %>%
  left_join(., acdongoingiw, by = c("ccode2", "year")) %>%
  mutate(., acdiwyear = ifelse(is.na(acdiwyear), 0, acdiwyear)) %>%
  mutate(., year = year + 1) # this to create the lag for the join below

save(df.dyadyear, file = "_data/_processed/dyadyears.Rda")

# Prepping civil war/intervention data ------------------------------------

# civil war sample from UCDP
ucdpexternal <- read.dta13("_data/_raw/externaldisaggregated.dta")

# generate start year of civil war, with first year of observation as the start year
ucdpstart <- ucdpexternal %>% 
  group_by(., conflictID, locationid1) %>%
  rename(., year = ywp_year, ccode1 = locationid1) %>%
  summarise(year = min(year)) %>%
  ungroup() %>%
  mutate(., ucdponset = 1) %>%
  mutate(., ccode1 = ifelse(ccode1==340, 345, ccode1)) %>%
  mutate(., ccode1 = ifelse(ccode1==678, 679, ccode1))
# fixing yugoslavia and yemen so it'll merge with dyad year

# joining dyadyear and ucdpstart
# need to see which ones do not make it

df.anti <- anti_join(ucdpstart, df.dyadyear, by = c("ccode1", "year"))
# yugoslavia and yemen causing trouble, yet again. going to manually recode

df.int <- left_join(ucdpstart, df.dyadyear, by = c("ccode1", "year"))

# intervener sample from UCDP
ucdpint <- ucdpexternal %>% filter(., external_alleged==0 & external_nameid<1000 & external_exists==1) %>%
  mutate(govrec = ifelse(str_detect(ywp_name, "^Government"), 1, 0)) %>%
  mutate(rebrec = ifelse(govrec==0, 1, 0)) %>%
  mutate(., govsupport = ifelse(govrec ==1 & external_type_X == 1, 3,
                                ifelse(govrec ==1 & external_type_L == 1 & external_type_Y==0 & external_type_W==0 & external_type_M==0 & external_type_T==0 & external_type_==0 & external_type_I==0 & external_type_O==0 & external_type_U==0 & external_type_X==0, 1,
                                       ifelse(govrec ==1 & (external_type_Y==1 | external_type_W==1 | external_type_M==1 | external_type_T==1 | external_type_==1 | external_type_I==1 | external_type_O==1 | external_type_U==1), 2, 0)))) %>%
  mutate(., govsupportyear = ifelse(govrec==1, ywp_year, NA)) %>%
  mutate(., rebsupport = ifelse(rebrec ==1 & external_type_X == 1, 3,
                                ifelse(rebrec ==1 & external_type_L == 1 & external_type_Y==0 & external_type_W==0 & external_type_M==0 & external_type_T==0 & external_type_==0 & external_type_I==0 & external_type_O==0 & external_type_U==0 & external_type_X==0, 1,
                                       ifelse(rebrec ==1 & (external_type_Y==1 | external_type_W==1 | external_type_M==1 | external_type_T==1 | external_type_==1 | external_type_I==1 | external_type_O==1 | external_type_U==1), 2, 0)))) %>%
  mutate(., rebsupportyear = ifelse(rebrec==1, ywp_year, NA)) %>%
  subset(., select = c("conflictID", "external_nameid", "govrec", "rebrec", "govsupport", "rebsupport", "govsupportyear", "rebsupportyear")) %>%
  rename(., ccode2 = external_nameid) %>%
  group_by(conflictID, ccode2) %>%
  summarise(
    across(c(1, 2, 3, 4), max, na.rm= TRUE),
    across(c(5, 6), min, na.rm = TRUE)
  ) %>%
  mutate(., govsupportyear = ifelse(govsupportyear==Inf, NA, govsupportyear)) %>%
  mutate(., rebsupportyear = ifelse(rebsupportyear==Inf, NA, rebsupportyear))

df.int <- left_join(df.int, ucdpint, by = c("conflictID", "ccode2"))

# check missing: not sure of what
missingdyad <- anti_join(ucdpint, df.int, by = c("conflictID", "ccode2"))


# Generating DV -----------------------------------------------------------

df.int <- df.int %>%
  filter(., year>=1975 & year<2010) %>%
  mutate(govrec = replace_na(govrec, 0)) %>%
  mutate(rebrec = replace_na(rebrec, 0)) %>%
  mutate(bothsides = ifelse(govrec==1 & rebrec==1, 1, 0)) %>%
  mutate(intervention = ifelse(govrec==1 & rebrec==0, 1, 
                               ifelse(govrec==0 & rebrec==1, 2, 
                                      ifelse(bothsides==1, 3, 0)))) %>%
  mutate(intervention = ifelse(bothsides==1 & govsupportyear<rebsupportyear, 1, 
                               ifelse(bothsides==1 & rebsupportyear<govsupportyear, 2, intervention)))


# Generating some IVs -----------------------------------------------------

# rescaling p2 and p2dist, so easier to compare with other vars
df.int$polity21 <- rescale(df.int$polity21, to = c(0,1))
df.int$polity22 <- rescale(df.int$polity22, to = c(0,1))

# now creating IVs
df.int <- df.int %>% 
  mutate(polydist = abs(v2x_polyarchy1-v2x_polyarchy2)) %>%
  mutate(libdist = abs(v2x_liberal1-v2x_liberal2)) %>%
  mutate(libdemdist = abs(v2x_libdem1-v2x_libdem2)) %>%
  mutate(p2dist = abs(polity21-polity22)) %>%
  mutate(opendist = abs(kaopen1-kaopen2)) %>%
  mutate(p2polydist = abs(p2dist - polydist))

# geo into dummies
df.int <- dummy_cols(df.int, select_columns = "geo")


# Saving ------------------------------------------------------------------

save(df.int, file = "_data/_processed/intervention.Rda")

write.dta(df.int, "_data/_processed/interdemo.dta")
