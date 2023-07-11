library(reshape2)

#input data
geo_info <- read.csv("avg-household-size.csv")
cancer_info <- read.csv("cancer_reg.csv")
#join dfs, create numeric df "data for corr"
cancer_info$county <- str_split_fixed(cancer_info$geography, ",", 2)[,2]
data_for_corr <- cancer_info %>% 
  select(!c("binnedinc", "geography","county"))
cancer_for_python <- cancer_info %>% 
  left_join(geo_info, by = 'geography') 
write.csv(cancer_for_python, "cancer_for_python_df.csv")

#get average by county and plot
state_aves <- cancer_info %>% group_by(county) %>%
  summarize(ave = mean(target_deathrate),
            std = sd(target_deathrate),
            the_sum = sum(target_deathrate),
            count = n())

plot <- ggplot(state_aves %>% top_n(30, ave))+
  geom_bar(stat = 'identity', fill = 'darkblue',
           aes(x = reorder(county, -ave), y = ave))+
  geom_errorbar(aes(x=reorder(county, -ave), ymin=ave-std, ymax=ave+std), width=0.4, colour="orange", alpha=0.9, size=1.3)+
  xlab("state")+ylab("average death-rate")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 70, vjust = 1, hjust=1),
        text = element_text(size = 20),
        legend.position = "none")
tiff("ave_deathrate_state.tiff", units="in", width=16, height=9, res=300)
print(plot)
dev.off()

#used feature importance from python models to look at trends in the data
plot <- ggplot(cancer_info_1 %>% 
                 filter(county %in% c(" Utah", " Kentucky")),
               aes(x = pctbachdeg25_over, y = target_deathrate))+
  geom_point(size = 10, aes(color = county))+
  ylim(0,300)+ ylab("death-rate") + xlab("% over 25 with bachelor degree")+
  theme_bw()+
  guides(color=guide_legend(title="State"))+
  theme(text = element_text(size = 30))+
  stat_poly_line() +
  stat_poly_eq(use_label(c("eq", "R2")), 
               label.x = 'left', label.y = 'bottom', size = 13)
tiff("death_by_bach_deg_ken_utah_lm.tiff", units="in", width=16, height=9, res=300)
print(plot)
dev.off()  