## analysis for interdemo

library(tidyverse)
library(ggplot2)
library(nnet)
library(sandwich)
library(lmtest)
library(stargazer)
library(mlogit)
library(scales)
library(expss)
library(car)
library(modelsummary)
library(Hmisc)
library(vtable)
library(AICcmodavg)
library(MNLpred)
library(patchwork)

# gonna drop those that intervene on both sides
df.int.noboth <- filter(df.int, intervention!=3)

# creating some labels and levels
# reordering here so it's easier to compare gov support (base) to reb support
df.int.noboth$intervention <- factor(df.int.noboth$intervention,
                                     levels = c(1, 0, 2), 
                                     labels = c("Gov. support", "No intervention", "Rebel support"))

df.int.noboth$geo <- factor(df.int.noboth$geo, 
                            levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9),
                            labels = c("W. Europe and USA", "E. Europe and C. Asia", "Middle East", "N. Africa", "S. and E. Asia", "Latin America", "Oceania", "Sub-Saharan", "Mexico and C. America"))

# Descriptive statistics --------------------------------------------------

sumtable(df.int.noboth, vars = c('p2dist', 'polydist', 'libdist', 'libdemdist', 'opendist', 'kappavv'), out = 'latex', file = '_output/_tables/desc.tex')

# Models ------------------------------------------------------------------

# models
p2model1 <- multinom(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2, data = df.int.noboth)
summary(p2model1)

p2model2 <- multinom(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.noboth, Hess = TRUE)
summary(p2model2)

polymodel1 <- multinom(intervention ~ polydist + v2x_polyarchy1 + v2x_polyarchy2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2, data = df.int.noboth)
summary(polymodel1)

polymodel2 <- multinom(intervention ~ polydist + v2x_polyarchy1 + v2x_polyarchy2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.noboth)
summary(polymodel2)

libmodel1 <- multinom(intervention ~ libdist + v2x_liberal1 + v2x_liberal2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2, data = df.int.noboth)
summary(libmodel1)

libmodel2 <- multinom(intervention ~ libdist + v2x_liberal1 + v2x_liberal2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.noboth)
summary(libmodel2)

libdemmodel1 <- multinom(intervention ~ libdemdist + v2x_libdem1 + v2x_libdem2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2, data = df.int.noboth)
summary(libdemmodel1)

libdemmodel2 <- multinom(intervention ~ libdemdist + v2x_libdem1 + v2x_libdem2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.noboth)
summary(libdemmodel2)

openmodel1 <- multinom(intervention ~ opendist + kaopen1 + kaopen2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2, data = df.int.noboth)
summary(openmodel1)

openmodel2 <- multinom(intervention ~ opendist + kaopen1 + kaopen2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.noboth)
summary(openmodel2)

kappamodel1 <- multinom(intervention ~ kappavv + v2x_polyarchy1 + v2x_polyarchy2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2, data = df.int.noboth)
summary(kappamodel1)

kappamodel2 <- multinom(intervention ~ kappavv + v2x_polyarchy1 + v2x_polyarchy2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.noboth)
summary(kappamodel2)

# joint table

stargazer(p2model2, polymodel2, libmodel2, libdemmodel2, type = "text", out = "_output/_tables/maintable1.tex")

stargazer(openmodel2, kappamodel2, type = "text", out = "_output/_tables/maintable2.tex")

# separate tables

stargazer(p2model1, p2model2, type="text", out = "_output/_tables/p2model.tex")

stargazer(polymodel1, polymodel2, type="text", out = "_output/_tables/polymodel.tex")

stargazer(libmodel1, libmodel2, type="text", out = "_output/_tables/libmodel.tex")

stargazer(libdemmodel1, libdemmodel2, type="text", out = "_output/_tables/libdemmodel.tex")

stargazer(openmodel1, openmodel2, type="text", out = "_output/_tables/openmodel.tex")

stargazer(kappamodel1, kappamodel2, type="text", out = "_output/_tables/kappamodel.tex")

models <- list(p2model1, p2model2, polymodel1, polymodel2, libmodel1, libmodel2, libdemmodel1, libdemmodel2, openmodel1, openmodel2, kappamodel1, kappamodel2)

model.names <- c('p2model1', 'p2model2', 'polymodel1', 'polymodel2', 'libmodel1', 'libmodel2', 'libdemmodel1', 'libdemmodel2', 'openmodel1', 'openmodel2', 'kappamodel1', 'kappamodel2')

aictab(cand.set = models, modnames = model.names)

# Graphs ------------------------------------------------------------------

# let's look at the data

cols <- c("#F76D5E", "#FFFFBF", "#72D8FF")

# creating a DF with numerical vars only
df.corr <- df.int.noboth %>% select(-c("conflictID", "ccode1", "ccode2", "year", "intervention", "rivalryname", "region", "type1", "type2", "type3", "geo"))

cor(df.corr)

corrmatrix <- rcorr(as.matrix(df.corr))
corrmatrix

corr <- as.data.frame(corrmatrix[[1]])
corr <- tibble::rownames_to_column(corr, "var1")
df.corr <- corr %>% pivot_longer(cols = -'var1', names_to='var2', values_to='correlation')

corr_p <- as.data.frame(corrmatrix[[3]])
corr_p <- tibble::rownames_to_column(corr_p, "var1")
df.corr_p <- corr_p %>% pivot_longer(cols = -'var1', names_to='var2', values_to='p')

df.corr.full <- left_join(df.corr, df.corr_p, by = c('var1', 'var2')) %>%
  drop_na() %>%
  mutate(corr_abs = abs(correlation)) %>%
  arrange(., desc(corr_abs))
# this yields a very large df

varlist <- c('p2dist', 'polity21', 'polity22', 'mindist', 'ongoingrivalry', 'cowmaj1', 'cowmaj2', 'wbgdp2011est1', 'wbgdp2011est2', 'wbpopest1', 'wbpopest2', 'wbgdppc2011est1', 'wbgdppc2011est2', 'upop_mc_1', 'upop_mc_2', 'cinc_mc_1', 'cinc_mc_2', 'growth_wdi_1', 'growth_wdi_2', 'acdcwyear', 'acdiwyear')

df.corr.vars <- df.corr.full %>%
  filter(., df.corr.full$var1 %in% varlist & df.corr.full$var2 %in% varlist)
# suggests some risk of multicollinearity

## graphing distribution of distance measures by conflictID

df.graph <- df.int %>% filter(., intervention!=3)

# creating some labels and levels
df.graph$intervention <- factor(df.graph$intervention,
                                     levels = c(0, 1, 2), 
                                     labels = c("No intervention", "Gov. support", "Rebel support"))


# creating density graphs for the explanatory variables
# setting list of explanatory variables
target_variables <- c('p2dist', 'polydist', 'libdist', 'libdemdist', 'opendist', 'kappavv')

# generating each density graph
density_list <- lapply(target_variables, function(each_variable) {
  ggplot(df.graph, aes(x = !!sym(each_variable), fill = factor(intervention))) +
    geom_density(alpha = 0.7) +
    scale_fill_manual(values = cols, name = "Intervention")
})

# putting the graphs together
density_graphs <- patchwork::wrap_plots(density_list, ncol = 3, guides = 'collect') &
  theme(legend.position = 'bottom')
ggsave(file = "_output/_figures/density.png", density_graphs, width = 24, height = 18, dpi = 300)

# let's take a look at this second peak of gov support
govoutliers <- filter(df.graph, p2dist>0.65 & intervention=="Gov. support" & polity22>0.6 & polity21<0.6)

govoutliers2 <- filter(df.graph, p2dist>0.65)

# Various graphs ----------------------------------------------------------

# graphing out the various explanatory variables, by themselves and by distance

# looking at distribution of polydist by conflictID

ggplot(df.graph, aes(x = mindist, fill = factor(intervention))) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values = cols, name = "Intervention", labels = c("No intervention", "Gov. support", "Rebel support"))

# let's look at distance

ggplot(df.graph, aes(x = mindist, fill = factor(intervention))) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values = cols, name = "Intervention", labels = c("No intervention", "Gov. support", "Rebel support"))
# it's easier to support governments than rebels far away, for logistical reasons!

# distribution of polydist by outcome

ggplot(df.graph, aes(x = polydist, fill = factor(intervention))) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values = cols, name = "Intervention", labels = c("No intervention", "Gov. support", "Rebel support"))

# let's look at polydist and p2dist together

ggplot(df.graph, aes(x = polydist, y = p2dist, color= factor(intervention))) +
  geom_point() + 
  scale_fill_manual(values = cols, name = "Intervention", labels = c("No intervention", "Gov. support", "Rebel support"))

ggplot(df.graph, aes(x = polydist, y = p2dist)) +
  geom_point(aes(color= factor(intervention))) +
  guides(colour = guide_legend(title= "Intervention")) +
  scale_color_discrete(labels = c("No intervention", "Gov. support", "Rebel support")) +
  facet_wrap("intervention")
# this seems to show that there are a lot of rebel-sided interventions with high p2dist but low polydist

inconsistent <- filter(df.graph, intervention==2 & p2dist>10 & polydist <0.25)

# now look at inconsistencies between the p2 and polyarchy as a function of distance

ggplot(df.graph, aes(x = p2polydist, y = mindist)) +
  geom_point(aes(color= factor(intervention))) +
  guides(colour = guide_legend(title= "Intervention")) +
  scale_color_discrete(labels = c("No intervention", "Gov. support", "Rebel support")) +
  facet_wrap("intervention")

ggplot(df.graph, aes(x = p2polydist, y = mindist)) +
  geom_point(aes(color= factor(geo))) +
  guides(colour = guide_legend(title= "Region")) +
  scale_color_discrete(labels = c("W. Europe and USA", "E. Europe and C. Asia", "Middle East", "N. Africa", "S. and E. Asia", "Latin America", "Oceania", "Sub-Saharan", "Mexico and C. America", "NA")) +
  facet_wrap("geo")

ggplot(df.graph, aes(x= polydist, fill = intervention)) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values = cols) +
  facet_wrap("geo")


# Maps --------------------------------------------------------------------

library(cshapes)
library(sf)

# need df with civil wars, interveners, colors for recipient and supporter, coordinates for dyad members, and lines

# getting a shape file for world map in 1977
cshp1977 <- cshp(date = as.Date("1977-1-1"), useGW = FALSE, dependencies = FALSE)
str(cshp1977)

# create df for civil war
df.int1977.cw <- df.int %>%
  filter(., year==1977) %>%
  select(., ccode1) %>%
  mutate(cw=1) %>%
  rename(cowcode=ccode1)

# create df for interveners
df.int1977.int <- df.int %>%
  filter(., year==1977 & intervention>=1) %>%
  select(., ccode2, intervention) %>%
  mutate(govsupport = ifelse(intervention==1, 1, 0)) %>%
  mutate(rebsupport = ifelse(intervention==2, 1, 0)) %>%
  group_by(ccode2) %>%
  summarise_all(max) %>%
  rename(cowcode=ccode2)

# time to join
cshp1977 <- left_join(cshp1977, df.int1977.cw, by = c("cowcode")) %>%
  left_join(., df.int1977.int, by = c("cowcode"))

# getting coordinates to fix the join

capcoord <- data.frame(cowcode = cshp1977$cowcode,
                       caplong = cshp1977$caplong,
                       caplat = cshp1977$caplat)

capcoord <- unique(capcoord)

# creating dyadic df for 1977 with main vars
# lot of things wrong here: need to filter by intervention, and then fix join
df.int1977.dyad <- df.int %>%
  filter(., year==1977 & intervention>=1) %>%
  select(., ccode1, ccode2, intervention) %>%
  left_join(., capcoord, by = c("ccode1" = "cowcode")) %>%
  rename(caplong_d = caplong, caplat_d = caplat) %>%
  left_join(., capcoord, by = c("ccode2" = "cowcode")) %>%
  rename(caplong_t = caplong, caplat_t = caplat)
  
# need long and lat
# need to just join data for that

plot(st_geometry(cshp1977))
cshp1977[is.na(cshp1977)] <- 0

cshp1977 <- cshp1977 %>%
  mutate(role = ifelse(cw==1, 1, ifelse(govsupport==1, 2, ifelse(rebsupport==1, 3, 0))))

levels(cshp1977$role) <- c("Civil war", "Gov. supporter", "Rebel supporter")

cols_map <- c("gray", "#F76D5E", "#FFFFBF", "#72D8FF")


# good start! just need to work on color and direction now for geom_curve
ggplot(data = cshp1977) +
  geom_sf(aes(fill = as.factor(role))) +
  scale_fill_manual(values = cols_map, name = "Intervention", labels = c("Neutral", "Civil war", "Gov. supporter", "Rebel supporter")) +
  geom_curve(data = df.int1977.dyad, 
             aes(x = caplong_t, y = caplat_t, xend = caplong_d, yend = caplat_d),
           curvature = -0.2,
           arrow = arrow(length = unit(0.1, "cm"), ends = "last"),
           size = 1,
           colour = "cadetblue4",
           alpha = 0.8)

# map for all civil wars with intervention: TBD


# Alt map stuff -----------------------------------------------------------


# library(maps)
# world <- map_data("world")
# str(world)
# 
# world <- left_join(world, df.int1977.cw, by = c("group" = "cowcode")) %>%
#   left_join(., df.int1977.int, by = c("group" = "cowcode")) %>%
#   world[is.na(world)] <- 0
# 
# ggplot(world, aes(x = long, y = lat, group = group)) + # need to fill in cw and int countries
#   geom_polygon(aes(fill = as.factor(cw)))
# 


