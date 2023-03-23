library("reticulate")
source_python("read_pickle.py")
df.fMRI<-read_pickle_file('fMRI_withscores.pkl')
np<-import('numpy')
age_nobias<-np$load('age_corrected.npy')
age_delta<-age_nobias-df.fMRI[,'Age']

unique(df.fMRI$SITE)
# for (site in unique(df.fMRI$SITE)){
#   print(site)
#   print(min(df.fMRI[df.fMRI$SITE==site,'Age']))
#   print(max(df.fMRI[df.fMRI$SITE==site,'Age']))
#   print(median(df.fMRI[df.fMRI$SITE==site,'Age']))
#   print(mean(df.fMRI[df.fMRI$SITE==site,'Age']))
#   print(sd(df.fMRI[df.fMRI$SITE==site,'Age']))
# }

variable<-c('Digit_Span_Forward','TMT_A','DSST', 'TMT_B','Digit_Span_Backward','BNT','ANI_Fluency','Category_Fluency','Letter_Fluency','VEG_Fluency',
            'CVLT_IM', 'CVLT_Long','MMSE','MOCA','WM_SUM','Systole','Diastole','BMI')


library(lme4)
for (var in variable){
  df.fMRI[,var]=as.numeric(df.fMRI[,var])
  if(var=='DSST'){
    index<-which(df.fMRI[,var]!='NaN', arr.ind=TRUE)
    
  }
  else if(var=='BNT'){
    index<-which(df.fMRI$SITE=='BLSA-3T' & df.fMRI[,var]!='NaN', arr.ind=TRUE)
  }
  else if (var=='ANI_Fluency'){
    index<-which(df.fMRI$SITE=='ABC' & df.fMRI[,var]!='NaN', arr.ind=TRUE)
    
  }
  else if (var=='TMT_B'){
    index<-which(df.fMRI[,var]!='NaN' & df.fMRI[,var]<400, arr.ind=TRUE)
  }
  else if (var=='Digit_Span_Forward'){
    index<-which(df.fMRI[,var]!='NaN' & df.fMRI[,var]>0, arr.ind=TRUE)
    #print(index)
  }
  else {
    index<-which(df.fMRI[,var]!='NaN', arr.ind=TRUE)
  }
  
  
  var_value<-df.fMRI[index,var]
  Age<-df.fMRI[index,'Age']
  Sex<-as.factor(df.fMRI[index,'Sex'])
  Site<-as.factor(df.fMRI[index,'SITE'])
  df.temp<-data.frame(Site,Age,Sex,var_value)
  #print(str(df.temp))
  if (length(unique(Site))>1){
    mixed.lmer<-lmer(formula=var_value ~ (1|Site) + Sex+ Age, data=df.temp)
  }
  else{
    mixed.lmer<-lm(formula=var_value ~  Age + Sex, data=df.temp)
  }
  var.after<-resid(mixed.lmer)
  
  age.delta<-age_delta[index]
  res<-cor.test(var.after,age.delta,method='pearson')
  print(var)
  print(round(res$estimate,4))
  print(round(res$p.value,4))
  #print(-log10(res$p.value))
  print(p.adjust(res$p.value,method='fdr',n=18))
  print('--------------------------------------------')
}


