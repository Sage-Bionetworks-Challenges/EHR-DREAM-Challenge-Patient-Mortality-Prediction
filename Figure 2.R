library(ggplot2)
library(gridExtra)

data <- read.csv("data/Table_2_Data.csv")

## Set team order 
data$Team <- factor(data$Team, levels=c('UW-biostat','Ivanbrugere','ProActa','AMbeRland','DMIS_EHR','PnP_India',
                                        'ultramangod671','HELM','Georgetown - ESAC','AI4Life','QiaoHezhe',
                                        'LCSB_LUX','chk','moore','tgaudelet'))

## Generate plot for the Area Under the Receiver Operator Curve changes from Leaderboard to Validation Stage. Includes error bars
p1 <- ggplot(data, aes(x=factor(Challenge.Stage), y=AUROC, group=Team, color=Team)) +
  geom_line() +
  geom_point(size=1.3) +
  geom_errorbar(aes(ymin=Low.95..CI, ymax=High.95..CI), width=.02) + 
  theme_classic() + 
  scale_color_manual(
    breaks=c('UW-biostat','Ivanbrugere','ProActa','AMbeRland','DMIS_EHR','PnP_India',
             'ultramangod671','HELM','Georgetown - ESAC','AI4Life','QiaoHezhe',
             'LCSB_LUX','chk','moore','tgaudelet'),
    values=c('#20B2AA','#FF4500','#B22222','#6666FF','#0080FF','#A9A9A9',
             '#A9A9A9','#A9A9A9','#A9A9A9','#A9A9A9','#A9A9A9',
             '#A9A9A9','#A9A9A9','#A9A9A9','#A9A9A9')) +
  scale_x_discrete(name ="Challenge Phase", labels=c("Leaderboard","Validation"), expand=c(0,0.3)) +
  ggtitle("Area Under the Receiver \n Operator Curve") + theme(plot.title = element_text(hjust = 0.5))


ggsave("AUROC_changes.png", p1, dpi=300)

## Generate plot for the Area Under the Precision Recall Curve changes from Leaderboard to Validation Stage. Error bars not included
p2 <- ggplot(data, aes(x=factor(Challenge.Stage), y=AUPR, group=Team, color=Team)) +
  geom_line() +
  geom_point(size=1.3) +
  theme_classic() + 
  scale_color_manual(
    breaks=c('UW-biostat','Ivanbrugere','ProActa','AMbeRland','DMIS_EHR','PnP_India',
             'ultramangod671','HELM','Georgetown - ESAC','AI4Life','QiaoHezhe',
             'LCSB_LUX','chk','moore','tgaudelet'),
    values=c('#20B2AA','#FF4500','#B22222','#6666FF','#0080FF','#A9A9A9',
             '#A9A9A9','#A9A9A9','#A9A9A9','#A9A9A9','#A9A9A9',
             '#A9A9A9','#A9A9A9','#A9A9A9','#A9A9A9')) +
  scale_x_discrete(name ="Challenge Phase", labels=c("Leaderboard","Validation"), expand=c(0,0.3)) +
  ggtitle("Area Under the Precision \n Recall Curve") + theme(plot.title = element_text(hjust = 0.5))

ggsave("AUPR_changes.png", p2, dpi=300)

grid.arrange(p1, p2, nrow=1)