library(Boruta)
library(reticulate)
pd <- import("pandas")

traindata <- pd$read_pickle("feat_select_master/fbook_CWI/camb_model/features/News_Dev_allInfo")

str(traindata)
names(traindata) <- gsub("_", "", names(traindata))

summary(traindata)

#split into x,y
#split into x,y
dataX <- traindata[, c("syllables", "length", "vowels", "pos", "dep num", "lemma", "synonyms", "hypernyms", "hyponyms", "wikipediafreq", 
"subtitlesfreq", "learnercorpusfreq", "complexlexicon", "bncfreq", "ogden", "simplewiki", "cald", "subimdb", "cnc", "img", "aoa",
"fam", "google frequency", "KFCAT", "KFSMP", "KFFRQ", "NPHN", "TLFRQ", "holonyms", "meronyms", "consonants", "learnersbigrams", "simplewikibigrams",
"ner", "googlecharbigram", "googlechartrigram", "simplewikifourgram", "learnerfourgram")]
dataY <- as.vector(traindata$complexbinary)

#perform boruta
set.seed(123)
boruta.train <- Boruta(dataX,dataY, doTrace = 2)
print(boruta.train)

par(mar=c(6,4,4,4)+.1)
plot(boruta.train, xlab = "", xaxt = "n",)
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i) boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)