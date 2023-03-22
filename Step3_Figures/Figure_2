####final code

library("reticulate")
library("gridExtra")
source_python("read_pickle.py")
df.age<-read_pickle_file('age_plot.pkl')
colnames(df.age)<-c('Chronological.age','Predicted.age.17','Predicted.age.25','Predicted.age.50','Predicted.age.75','Predicted.age.100','Predicted.age.125','Predicted.age.150','Concatenated','site')
library(tidyverse)
library(ggpubr)
K=c(17,25,50,75,100,125,150)
plots<-c()
corr<-c(0.51,0.59,0.63,0.60,0.59,0.66,0.73,0.79)
palette_test <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                  "#0072B2", "#D55E00", "#CC79A7")

for (i in 1:8){
  if (i==8){
    temp<-'Concatenated' 
    plot.i<-ggscatter(df.age, x='Chronological.age',y=temp,shape=20,color='site',palette=palette_test, xlab='Chronological age', ylab='Predicted age', alpha=0.5,size=3)+
      geom_smooth(method = "lm", se = FALSE)+
      geom_abline(intercept = 0, slope = 1, color="black", linetype="dashed", size=1)+
      #geom_text(family="Times New Roman",size=14)+
      geom_text(x=30, y=95.5, label=" Multiscale",size=7,family='Arial')+
      geom_text(x=30, y=91, label=paste("corr=",corr[i],sep="",collapse=""),size=7,family='Arial')+
      font("xlab", size = 20)+
      font("ylab", size = 20)+
      font("legend.title", size = 20)+
      font("legend.text", size = 20)+
      theme(legend.key.size = unit(0.2, 'cm'))
  } else
  {
    temp<-paste('Predicted.age.',K[i],sep="",collapse="")
    plot.i<-ggscatter(df.age, x='Chronological.age',y=temp,shape=20,color='site', palette=palette_test, xlab='Chronological age', ylab='Predicted age', alpha=0.5,size=1)+
      geom_smooth(method = "lm", se = FALSE)+
      geom_abline(intercept = 0, slope = 1, color="black", linetype="dashed", size=1)+
      #geom_text(family="Times New Roman",size=14)+
      geom_text(x=34.5, y=97, label=paste('    k=',K[i],sep="",collapse=""),size=7,family='Arial')+
      geom_text(x=34.5, y=91, label=paste("corr=",corr[i],sep="",collapse=""),size=7,family='Arial')+
      font("xlab", size = 20)+
      font("ylab", size = 20)+
      font("legend.title", size = 20)+
      font("legend.text", size = 20)+
      theme(legend.key.size = unit(0.2, 'cm'))
  }
  plot.i<-ggpar(plot.i, ylim = c(20, 100),xlim=c(20,100))
  
  plots[[i]]<-plot.i
}

a<-ggarrange(plots[[8]],ggarrange(plotlist=plots[1:4],ncol=2,nrow=2,legend='none',labels=c('B','C','D','E')),ncol=2,align='v',legend='none',labels=c('A'))
#ggsave("test4.png", dpi=300,width=20, height=8, units='in')

b<-ggarrange(plots[[5]],NULL,plots[[6]],NULL,plots[[7]],widths = c(1, 0.4, 1,0.4,1), nrow=1,align='v',legend='none',labels=c('F','','G','','H'))
#c<-grid.arrange(a,b,nrow=2)
library('cowplot')
c<-plot_grid(a,b, align = "v", nrow = 2, rel_heights = c(18/30, 12/30))
c<-plot_grid(a,b, align = "v", nrow = 2, rel_heights = c(2/3, 1/3))
ggsave("correlation_result.png", plot=c,dpi=500,width=16, height=12, units='in')