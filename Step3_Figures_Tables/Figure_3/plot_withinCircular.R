library(circlize)
library(REdaS)
library("reticulate")

source_python("read_pickle.py")
color=c('#78127c','#4682b4','#007672','#c43afa','#dcf8a4','#e69422','#cd3e4e')

connect<-read_pickle_file('connectmatrix.pkl') # it's a dictionary. each is a connectivity matrix (0-1)
node_project<-read_pickle_file('projection.pkl') #each row represents a node. It has been sorted according to the yeo 7 networks (network 1 to 7) so networks 
# belong to same yeo 7 networks can be clustered together.
np<-import('numpy')
draw<-np$load('draw.npy') # this is the matrix from python, column 1: network scale; column 2 and column 3: connectivity network nodes 
# column 4: the yeo 7 network column 2 belong to; column 5: the yeo 7 network column 3 belong to;

rgb2hex <- function(r,g,b) rgb(r, g, b, maxColorValue = 255)
c_n=matrix(c(120,18,124,70,130,180,0,118,114,196,58,250,220,248,164,230,148,34,205,62,78), nrow=7, byrow=TRUE) # the 7 color for the yeo 7 networks visulization

#K=c(17,50,75,100,125,150)

### the new result has no important features from 25 network
K=c(17,25,50,75,100,125,150)
#K=c(17, 50)
for (k in 1:1){
  mat<-connect[[k]]
  node<-node_project[[k]]
  matrix_name=character()
  grid.col=character()
  
  ### grid color is the color of the circular
  for (i in 1:dim(node)[1]){
    #matrix_name[i]=paste('net_',node[i,1]+1,sep="",collapse="")
    matrix_name[i]=paste(node[i,1]+1,sep="",collapse="")
    grid.col[i]=c(color[node[i,3]+1])
    #names(grid.col[i])=c(matrix_name[i])
  }
  #print(matrix_name)
  names(grid.col)=matrix_name
  rownames(mat)<-matrix_name
  colnames(mat)<-matrix_name
  #print(mat)
  
  #col_mat = matrix(, nrow = dim(node)[1], ncol = dim(node)[1])
  #col_mat = rand_color(dim(node)[1], transparency = 0.5)
  
  col_mat = rand_color(length(mat))
  dim(col_mat) = dim(mat)  # to make sure it is a matrix
  rownames(col_mat)<-matrix_name
  colnames(col_mat)<-matrix_name
  
  
  
  ###color the links
  for (j in 1:length(draw[draw[,1]==K[k],1])){
    index_row=draw[draw[,1]==K[k],2][j]+1
    index_col=draw[draw[,1]==K[k],3][j]+1
    c_1=mean(c(c_n[draw[draw[,1]==K[k],4][j]+1,1],c_n[draw[draw[,1]==K[k],5][j]+1,1]))
    c_2=mean(c(c_n[draw[draw[,1]==K[k],4][j]+1,2],c_n[draw[draw[,1]==K[k],5][j]+1,2]))
    c_3=mean(c(c_n[draw[draw[,1]==K[k],4][j]+1,3],c_n[draw[draw[,1]==K[k],5][j]+1,3]))
    #         print(c_1)
    #         print(c_2)
    #         print(c_3)
    col_mat[paste(index_row,sep="",collapse=""),paste(index_col,sep="",collapse="")]=rgb2hex(c_1,c_2,c_3)
    col_mat[paste(index_col,sep="",collapse=""),paste(index_row,sep="",collapse="")]=rgb2hex(c_1,c_2,c_3)
  }
  #print(col_mat)
  #     setEPS()
  #     postscript("test.eps")
  
  
  ## this is for grep the brain network images, they are the contributing features of networks in different scales
  output=paste('new_network_con_',K[k],'.pdf', sep="",collapse="")
  library(tidyverse)
  library(png)
  DATA = data.frame(images = list.files(paste("important_network/",K[k],sep="",collapse=""), full.names = T), stringsAsFactors = F) %>% 
    mutate(
      names = gsub("[a-zA-Z]|[[:punct:]]","", images),
      #values = sample(0:100, size=nrow(.), replace = T)
    )
  #print(DATA)
  pdf(file=output)
  #circos.par(start.degree=180)
  #chordDiagram(mat, symmetric=TRUE, grid.col=grid.col,col='grey',annotationTrack = c("name","grid"),order=matrix_name)
  
  #     chordDiagram(mat, grid.col = grid.col, col='grey',symmetric=TRUE, annotationTrack = "grid",order=matrix_name, preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
  #     circos.track(track.index = 1, panel.fun = function(x, y) {
  #     circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, 
  #         facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.8))
  # }, bg.border = NA)
  #print(col_mat)
  par(cex = 1.5, mar = c(0, 0, 0, 0),font=2)
  chordDiagram(mat, symmetric=TRUE, grid.col=grid.col,col=col_mat, transparency=0.25,annotationTrack =c('name','grid'),
               order=matrix_name,annotationTrackHeight=c(0.01, 0.05),preAllocateTracks = list(track.height=0.3))
  u=0
  for(si in get.all.sector.index()){
    #print(si)
    xplot=get.cell.meta.data("xplot",si)
    #print(xplot)
    u=u+1
    
    # Small workaround because coordinate 0 should be 360
    if(xplot[1] == 0) xplot[1] = 360
    
    x=.86*cos(deg2rad((xplot[2]+xplot[1])/2))
    y=.86*sin(deg2rad((xplot[2]+xplot[1])/2))
    temp=paste('IC_',si,'.png',sep="",collapse="")
    #print(temp)
    DATA$images[grep(temp, DATA$images)] %>%
      readPNG(info=T) %>% 
      rasterImage(x-0.07, y-0.07, x+0.07, y+0.07)
  }
  #title(paste("k=",K[k],"", sep="",collapse=""))
  circos.clear()
  dev.off()
}