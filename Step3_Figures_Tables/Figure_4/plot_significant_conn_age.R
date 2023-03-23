library("reticulate")
library('DescTools')
source_python("read_pickle.py")
df.conn_age<-read_pickle_file('conn_age.pkl')
library(ggpubr)
df.conn_age[,'SEX']<-as.factor(df.conn_age[,'SEX'])
df.conn_age[,'SITE']<-as.factor(df.conn_age[,'SITE'])
previous_name = colnames(df.conn_age)
#df.conn_age[,c(1:47)]=FisherZ(df.conn_age[,c(1:47)])
#str(df.conn_age)
num_fea=dim(df.conn_age)[2]-3


### control for the sex and site information
library(lme4)
for (i in 1:num_fea){
  colnames(df.conn_age)[i]<-paste('feature_',i,sep="")
} 
regress.cov<-function(i){
  mixed.lmer<-lmer(as.formula(paste(colnames(df.conn_age)[i],'~ SEX + (1|SITE)', sep="")) ,data=df.conn_age)
  df.conn_age[,i]=resid(mixed.lmer)
}
df.conn_age[,c(1:num_fea)]<-lapply(1:num_fea,regress.cov)



make_plot<-function(i){
  res <- cor.test(df.conn_age[,i], df.conn_age[,64], 
                  method = "pearson")
  if(res$p.value<0.05 && res$estimate>0){
    print(i)
    temp=paste(strsplit(previous_name[i]," ")[[1]][1],' Net: ',strsplit(previous_name[i]," ")[[1]][3],sep="",collapse="")
    A<-ggscatter(df.conn_age, x='Age',y=names(df.conn_age)[i],add='reg.line',ylab=temp,shape=21, size=1,conf.int=TRUE, add.params=list(color='red',fill='lightgray'))
    A<-ggpar(A,xlim=c(20,100),ylim=c(-1,1))+
      font("xlab", size = 20,face=2)+
      font("ylab", size = 20,face=2)+
      font("legend.title", size = 20,face=2)+
      font("legend.text", size = 20,face=2)
    A+stat_cor(method='pearson',label.x=25,label.y=-1,hjust=0,label.sep='\t',size=8,fontface='bold',color='red')
    
  }
  else if(res$p.value<0.05 && res$estimate<0){
    print(i)
    temp=paste(strsplit(previous_name[i]," ")[[1]][1],' Net: ',strsplit(previous_name[i]," ")[[1]][3],sep="",collapse="")
    A<-ggscatter(df.conn_age, x='Age',y=names(df.conn_age)[i],add='reg.line',ylab=temp,shape=21, size=1,conf.int=TRUE, add.params=list(color='blue',fill='lightgray'))
    A<-ggpar(A,xlim=c(20,100),ylim=c(-1,1))+
      font("xlab", size = 20,face=2)+
      font("ylab", size = 20,face=2)+
      font("legend.title", size = 20,face=2)+
      font("legend.text", size =20,face=2)
    A+stat_cor(method='pearson',label.x=25,label.y=-1,hjust=0,label.sep='\t',size=8,fontface='bold',color='blue')
    
  }
}

plots<-lapply(1:num_fea,make_plot)
plots<-plots[!sapply(plots,is.null)]
length(plots)


a<-ggarrange(plotlist=plots,ncol=5,nrow=3,align='v')
ggsave("conn_age_new_40_60.png", dpi=500,width=23, height=15, units='in')
#ggsave(file="conn_age_new.eps")