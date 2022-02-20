# C6 Project8 #############################################################

# 환경설정
rm(list=ls())
setwd('C:/rwork/')
options(max.print = 300)

# 라이브러리
#install.packages("tree")
library(tree)
#install.packages("doBy")
library(doBy)
#install.packages("party")
library(party) #ls("package:party")
#install.packages("caret")
library(caret) #ls("package:caret")
#install.packages("dplyr")
library(dplyr)
#install.packages("e1071") #나이브 베이즈
library(e1071)
#install.packages("randomForest")
library(randomForest)
#install.packages("data.table")
library(data.table)
#install.packages("xgboost")
library(xgboost)
#install.packages("plyr")
library(plyr)
#install.packages("car")#vif 공분산분석
library(car)
#install.packages("kernlab")
library(kernlab)
#install.packages("class")
library(class)
#install.packages("nnet")
library(nnet)
#install.packages('adabag')
library(adabag)
#install.packages('pROC')
library(pROC)
#install.packages("MASS") #ROC 커브
library(MASS)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("pls") #시계열분석
library(pls)


# spam 데이터 셋 ==============================================================
  
  # spam데이터 설명
  # 불러온 spam 데이터는 4601개의 이메일에서 등장하는 단어의 종류와 관련된 58개의 변수로 구성되어있다. 
  # 58개의 변수 중 처음 48개 변수(A.1~A.48)은 총 단어수 대비 해당 단어의 출현비율을 나타내며, 
  # 6개 변수(A.49~A.54)는 총 문자수 대비 특정 문자의 출현비율을 나타내며, 
  # 3개 변수(A.55~A.57)은 연속되는 대문자 철차의 평균길이, 최대길이, 대문자의 총수를 나타낸다. 
  # 마지막 변수(spam)스팸 메일의 여부를 타나냅니다. 
  # 즉 spam 변수가 종속변수가 되며 나머지 A.1~57 변수가 예측변수가 된다.. 
  # 결측값은 없으며 전체에서 스팸메일은 1813개다.



## 1.데이터 전처리 ============================================================

# 1)데이터 불러오기
spam <- read.csv('spam.csv', header = T)
str(spam)
set.seed(9999)

# 2)데이터 정규화
normalize <- function(x) {
  return ((x-min(x))/(max(x)-min(x)))
}
spam[56:57] <- normalize(spam[56:57])

# 3) train/test sets 생성
  #(1)doBy train/test sets 생성
  spam_train_doBy <- sampleBy(~spam, frac = 0.7, data = spam)
  enrow <- rownames(spam_train_doBy)
  nurow <- as.numeric(gsub('\\D','',enrow))
  spam_test_doBy <- spam[-nurow,]
  
  #(2)caret train/test sets 생성
  set.seed(8888)
  train_idx <- createDataPartition(spam$spam, p=0.7, list=F)
  spam_train_caret <- spam[train_idx,]
  spam_train_label_caret <- spam$spam[train_idx]
  spam_test_caret <- spam[-train_idx,]




## 2.분석 =====================================================================

#### 1)나이브 베이즈 ####
# e1071 패키지
  #(1)나이브베이즈 학습 모델 생성
  spam_nb_doby <- naiveBayes(spam ~ .,
                             data = spam_train_doBy,
                             laplace = 1)
  spam_nb_doby
  
  #(2)예측 분류 결과 생성
  spam_nb_pred_doby <- predict(spam_nb_doby, newdata = spam_test_doBy, type = 'class')
  
  #(3)나이브베이즈 적용 분류 결과 도출
  table(spam_nb_pred_doby, spam_test_doBy$spam)
  
  #(4)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(spam_nb_pred_doby, as.factor(spam_test_doBy$spam))
  
  ## e1071 패키지의 나이브베이즈는 약 72%의 정확도로 분류


# caret패키지
  #(1)모델 생성
  ctrl <- trainControl(method="cv", 10)
  spam_nb_caret <- train(spam ~ ., data = spam_train_caret, 
                         method = 'naive_bayes',
                         trControl = ctrl)
  spam_nb_caret
  
  #(2)예측 분류 결과 생성
  spam_nb_pred_caret <- predict(spam_nb_caret, newdata = spam_test_caret)
  
  #(3)나이브베이즈 적용 분류 결과 도출
  table(spam_nb_pred_caret, spam_test_caret$spam)
  
  #(4)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(spam_nb_pred_caret, as.factor(spam_test_caret$spam))
  
  
  ## caret의 나이브베이즈는 70%의 정확도로 분류




#### 2)SVM(서포트 벡터 머신) ####

  # 2.1 파라미터 최적값찾기
  # tune.svm(factor(spam) ~ ., data = spam, gamma = 2^(-1:1), cost = 2^(1:4))
  # 결과
  #  Parameter tuning of ‘svm’:
  #  - sampling method: 10-fold cross validation 
  #  - best parameters:
  #   gamma cost
  #    0.5    4
  #  - best performance: 0.16714 
  
# e1071 패키지
  #(1)SVM 학습 모델 생성
  spam_svm_doBy <- svm(factor(spam) ~ .,
                       data = spam_train_doBy,
                       gamma = 0.5,
                       cost = 4)
  spam_svm_doBy
  
  #(2)예측 분류 결과 생성
  spam_svm_pred_doBy <- predict(spam_svm_doBy, newdata = spam_test_doBy)
  spam_svm_pred_doBy
  
  #(3)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(spam_svm_pred_doBy, factor(spam_test_doBy$spam))
  
  # e1071패키지의 SVM은 정확도(0.8101). 즉, 약 81% 정확도로 분류하였다.



# caret패키지
  #(1)caret패키지 SVM훈련 모델 생성
  ctrl <- trainControl(method="cv", 10)
  spam_svm_caret <- train(spam ~ .,
                          data = spam_train_caret,
                          method = 'svmRadial',
                          trControl = ctrl,
                          tuneGrid = expand.grid(sigma= 0.5 , C = 4))
  
  #(2)caret패키지 SVM학습모델 예측 분류 결과 생성
  spam_svm_pred_caret<- predict(spam_svm_caret, newdata = spam_test_caret)
  
  #(3)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(spam_svm_pred_caret, factor(spam_test_caret$spam))
  
  # caret패키지 SVM은 정확도(0.8289). 즉, 83% 정확도로 분류하였다.

  


#### 3)로지스틱 회귀분석 ####
# stats 패키지
  #(1)glm함수 로지스틱 회귀분석 훈련모델 생성
  spam_glm_doBy <- glm(factor(spam) ~ ., data = spam_train_doBy, family = 'binomial')
  
  #(2)예측 분류 결과 생성
  spam_glm_pred_doBy <- predict(spam_glm_doBy, newdata = spam_test_doBy, type = 'response')
  
  #(3)모델 성능 평가 지표(정확도 확인)
  spam_glm_pred_doBy2 <- ifelse(spam_glm_pred_doBy < 0.5, 'spam', 'email')  # 컷오프 0.5로 설정하여 사후확률이 0.5초과이면 spam, 05이하이면 email로 예측한다.
  confusionMatrix(factor(spam_glm_pred_doBy2), factor(spam_test_doBy$spam))
  
  # glm 로지스틱 회귀분석은 약 91% 정확도로 분류하였다.



# caret 패키지
  #(1)caret 패키지 로지스틱 회귀 훈련 모델 생성
  ctrl <- trainControl(method="cv", 10)
  spam_glm_caret <- train(factor(spam) ~ .,
                          data = spam_train_caret,
                          method = 'glm'
                          , trControl = ctrl)
  
  #(2)caret패키지 로지스틱 회귀 예측 분류 결과 생성
  spam_glm_pred_caret <- predict(spam_glm_caret, newdata = spam_test_caret, type = 'raw')
  
  #(3)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(factor(spam_glm_pred_caret), factor(spam_test_caret$spam))
  
  # caret패키지의 로지스틱 회귀분석은 약 91% 정확도로 분류하였다.




#### 4)최근접 이웃 모델(KNN) ####
# class 패키지
  #(1)doBy 데이터 셋 라벨링
  spam_train_label_doBy <- spam$spam[nurow] 
  spam_test_label_doBy <- spam$spam[-nurow]
  
  #(2)class패키지 knn학습모델 생성(k = 57)
  spam_knn_doBy <- knn(train = spam_train_doBy[,-58],
                       test = spam_test_doBy[,-58],
                       cl = spam_train_label_doBy,
                       k = 57)
  spam_knn_doBy
  
  #(3)class패키지 knn학습모델 분류 결과 도출
  tt <- table(spam_test_label_doBy, spam_knn_doBy)
  tt
  
  #(4)모델 성능 평가 지표(정확도 확인)
  sum(tt[row(tt) == col(tt)])/sum(tt)  # 정분류율
  1-sum(tt[row(tt) == col(tt)])/sum(tt)  # 오분류율
  
  
  # 정분류율(0.876)
  # 따라서 class패키지의 knn모델은 약 88%의 정확도로 분류함.

  

# caret 패키지
  #(1)caret패키지 knn학습모델 생성(k = 57)
  tune <- trainControl(method = 'cv', number = 10)
  spam_knn_caret <- train(spam ~ ., data = spam_train_caret,
                          method = 'knn',
                          tuneGrid = expand.grid(k=57),
                          trControl = tune)

  #(2)caret패키지 knn학습모델 분류 결과 도출
  spam_knn_pred_caret<- predict(spam_knn_caret, newdata = spam_test_caret)
  
  
  #(3)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(spam_knn_pred_caret, as.factor(spam_test_caret$spam))
  
  # caret패키지의 knn학습모델을 약 88% 정확도로 분류함.




#### 5)인공신경망 ####
#nnet 패키지
  #(1)nnet패키지 인공신경망 학습모델 생성
  spam_nnet_doBy <- nnet(factor(spam) ~ .,
                         data = spam_train_doBy,
                         size = 4,
                         decay = 5e-04)  # 가장 정확하다는 옵션 선택. 
  spam_nnet_doBy
  
  #(2)nnet패키지 인공신경망 학습모델 분류 결과 도출
  spam_nnet_pred_doBy <- predict(spam_nnet_doBy, newdata = spam_test_doBy, type = 'class')
  
  #(3)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(factor(spam_nnet_pred_doBy), factor(spam_test_doBy$spam))
  
  # nnet패키지 인공신경망은 95% 정확도로 분류함



#caret패키지
  #(1)caret패키지 인공 신경망 학습모델 생성
  spma_nnet_caret <- train(spam1 ~ .,
                           data = spam_train_caret,
                           method = 'nnet',
                           trace = F,
                           tuneGrid = expand.grid(.size= 4, .decay = 5e-04))
  
  #(2)caret패키지 인공 신경망학습모델 분류 결과 도출
  spam_nnet_pred_caret<- predict(spma_nnet_caret, newdata = spam_test_caret)
  
  #(3)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(spam_nnet_pred_caret, as.factor(spam_test_caret$spam))


  # caret패키지 인공신경망은 95% 정확도로 분류함

women


# abalone 데이터 셋 ===========================================================

# 설명)
# Sex		          nominal			M, F, and I (infant)
# Length		      continuous	mm	Longest shell measurement
# Diameter	      continuous	mm	perpendicular to length
# Height		      continuous	mm	with meat in shell
# Whole weight	  continuous	grams	whole abalone
# Shucked weight	continuous	grams	weight of meat
# Viscera weight	continuous	grams	gut weight (after bleeding)
# Shell weight	  continuous	grams	after being dried
# Rings		        integer			+1.5 gives the age in years


## 1.데이터 전처리 ============================================================

# 1)데이터 셋 불러오기
data1 <- read.csv('abalone.csv',header = F)
names(data1) <- c('Sex','Length','Diameter','Height','WholeWeight',
                  'ShuckedWeight','VisceraWeight','ShellWeight','Rings')

# 2)데이터 추출/자료형 변환
abalone <- subset(data1, Sex != 'I') #유아기의 전복은 제외
abalone <- droplevels(abalone)
abalone$Sex <- as.factor(abalone$Sex)
sapply(abalone,class)

# 3) train/test sets 생성
  #(1)doBy train/test sets 생성
  set.seed(1111)
  abalone_doBy_train <- sampleBy(~Sex, frac=0.7, data=abalone) #전복의 성별을 기준으로 동일한 비율로 나눔
  abalone_doBy_test <- sampleBy(~Sex, frac=0.3, data=abalone)
  
  #(2)caret train/test sets 생성
  set.seed(1000)
  intrain <- createDataPartition(y=abalone$Sex, p=0.7, list=FALSE) 
  abalone_caret_train <-abalone[intrain, ]
  abalone_caret_test <-abalone[-intrain, ]
  table(abalone_caret_train$Sex)
  table(abalone_caret_test$Sex)




## 2.분석 ====================================================================  

### 상황1) ####
#귀무가설: 각종 전복의 수치로 성별을 오차범위 5%내로 구분할 수 없다.
#대립가설: 각종 전복의 수치로 성별을 오차범위 5%내로 구분할 수 있다.

# 종속변수가 범주형이고 설명변수가 연속형의 자료이므로,
# 의사결정나무에서는 CART, C5.0, QUEST기법이 활용 가능하다.

#### 1)의사결정나무 ####
# party 패키지
  #(1)학습모델 생성
  treeOption1 <- ctree_control(maxdepth = 10)
  abalone_tree1 <- ctree(Sex~.,
                         data = abalone_doBy_train,
                         controls = treeOption1)
  plot(abalone_tree1, compress=TRUE)
  
  #(2)예측치 생성
  table(abalone_doBy_train$Sex, predict(abalone_tree1,data=abalone_doBy_train),dnn = c('Actual','Predicted'))
  predict(abalone_tree1,data=abalone_doBy_train)
  
  #(3)모형의 정확성 검정
  confusionMatrix(data=abalone_doBy_test$Sex,predict(abalone_tree1,abalone_doBy_test))
  
  #정확도 약54%



# caret 패키지
  #(1)모델 생성 및 시각화
  treemod <- train(Sex ~., method = "ctree", data=abalone_caret_train)
  plot(treemod)
  
  #(2)예측 및 모델 평가
  pred = predict(newdata=abalone_caret_test,treemod)
  table(pred)
  
  #(3)모델 평가
  confusionMatrix(pred, abalone_caret_test$Sex)
  
  #정확도 약54%
  
  # 결론: 전복의 성별은 전복의 각종 수치로는 구별이 불가능하므로 대립가설을 기각한다.




### 상황2) ####
# 전복의 각종 수치가 Rings의 크기에 미치는 영향을 파악한다.

#### 2) 다중회귀분석 ####
#stats 패키지
  #(1)학습모델 생성
  abalone_lm_model <- lm(Rings ~., data=abalone_doBy_train)
  summary(abalone_lm_model) #p-값 확인: 0.05이하이므로 독립변수들 간의 모형은 유의하다.
  vif(abalone_lm_model)
  
  #다중공선성 문제가 가장 심각한 변수를 제외한다.
  abalone_lm_model2 <- lm(Rings ~ ., data = abalone_doBy_train[,-5])
  summary(abalone_lm_model2)
  
  #결과 해석
  # 성별의 p-값이 다소 높지만 큰 영향을 주는 변수는 아니기 때문에 무시하기로 한다.
  # 그 외에 Rings에 가장 큰영향을 미치는 변수는 Shellweight, Diameter, shuckedWeight
  # 순서로 영향을 미쳤다.

#caret 패키지
  #(1) 학습모델 생성
  ctrl <- trainControl(method="cv", 10)
  abalone_lm_caret <- train(Rings ~ .,
                            data = abalone_caret_train[,-5],
                            na.action = na.omit,
                            method = 'lm',
                            trControl = ctrl)
  summary(abalone_lm_caret)
  #결과 해석
  # 그 외에 Rings에 가장 큰영향을 미치는 변수는 Shellweight, Height, shuckedWeight
  # 순으로 파악됐다.




# titanic 데이터 셋 ===========================================================

# 설명)
# pclass :    1, 2, 3등석 정보를 각각 1, 2, 3으로 저장
# survived :  생존 여부. survived(생존=1), dead(사망=0)
# name :      이름(제외)
# sex :       성별. female(여성), male(남성)
# age :       나이
# sibsp :     함께 탑승한 형제 또는 배우자의 수
# parch :     함께 탑승한 부모 또는 자녀의 수
# ticket :    티켓 번호(제외)
# fare :      티켓 요금
# cabin :     선실 번호(제외)
# embarked :  탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)
# boat     :  (제외)Factor w/ 28 levels "","1","10","11",..: 13 4 1 1 1 14 3 1 28 1 ...
# body     :  (제외)int  NA NA NA 135 NA NA NA NA NA 22 ...
# home.dest:  (제외)


## 1.데이터 전처리 ============================================================

# 1)데이터 셋 불러오기
data2 <- read.csv('titanic.csv',header = T)
titanic <- data2[,c(2,4,5,11)]
head(titanic)
colnames(titanic)

sapply(titanic,class)  #데이터 자료형 확인
shapiro.test(na.omit(titanic$age))  # 정규성 검정
# 0.05보다 작으므로 정규분포가 아니다.

# 2) train/test sets 생성
  #(1)doBy train/test sets 생성
  #샘플링
  set.seed(2124)
  titanic_doBy_train <- sampleBy(~survived, frac=0.7, data=titanic) #생존여부를 기준으로 동일한 비율로 나눔
  titanic_doBy_test <- sampleBy(~survived, frac=0.3, data=titanic)
  #자료형 변환
  titanic_doBy_train$survived <- as.factor(titanic_doBy_train$survived)
  titanic_doBy_test$survived <- as.factor(titanic_doBy_test$survived)
  #NA값 제거
  titanic_doBy_train <- subset(titanic_doBy_train,!is.na(titanic_doBy_train$age))
  titanic_doBy_test <- subset(titanic_doBy_test,!is.na(titanic_doBy_test$age))
  
  #(2)caret train/test sets 생성
  #샘플링
  set.seed(2223)
  intrain <- createDataPartition(y=titanic$survived, p=0.7, list=FALSE)
  titanic_caret_train <-titanic[intrain, ]
  titanic_caret_test <-titanic[-intrain, ]
  #자료형 변환
  titanic_caret_train$survived <- as.factor(titanic_caret_train$survived)
  titanic_caret_test$survived <- as.factor(titanic_caret_test$survived)
  #NA값 제거
  titanic_caret_train <- na.omit(titanic_caret_train)
  titanic_caret_test <- na.omit(titanic_caret_test)




## 2.분석 =====================================================================

### 상황1 ####
# 생존자들의 평균나이가 28.92세 일 때, 사망한 사람들의 평균나이와 차이가 있는지 
# 검정하기 위해 사망한 사람들을 랜덤으로 선정하여 검정을 시행한다.

#귀무가설: 생존한 사람과 생존하지 못한 사람은 나이의 평균에 차이가 없다.
#대립가설: 생존한 사람과 생존하지 못한 사람은 나이의 평균에 차이가 있다.

# 종속변수가 연속형이고 설명변수가 범주형의 자료이므로,
# 통계기반의 T-test 기법이 활용 가능하다.

#### 1)two sample T-test ####
# stats 패키지
  #(1)대응하는 두 집단 생성
  dead <- subset(titanic_doBy_train,titanic_doBy_train$survived == 0)
  
  #(2)양측 검정 - titanic객체의 기존 모집단의 평균 28.92세 비교
  t.test(dead$age, mu = 28.92)
  qqnorm(dead$age)
  qqline(dead$age, lty = 1, col = "blue")
  t.test(dead$age, mu = 28.92, alter = "two.side", conf.level = 0.95)
  
  #p-값이 유의수준 0.05보다 낮기 때문에 평균 수명에 차이가 있다고 볼 수 있다.
  
  #(3)단측 검정 - 방향성을 가짐
  t.test(dead$age, mu = 28.92, alter= "greater", conf.level = 0.95)
  
  #(4)귀무가설의 임계값 계산
  qt(0.05,427,lower.tail = F)
  
  #귀무가설을 기각할 수 있는 임계값 = 1.64843
  #검정통계량 t=2.0755, 유의확률P=0.01927



# caret 패키지
  #(1)대응하는 두 집단 생성
  dead2 <- subset(titanic_caret_train,titanic_caret_train$survived == 0)
  
  #(2)양측 검정 - titanic객체의 기존 모집단의 평균 28.92세 비교
  t.test(dead2$age, mu = 28.92)
  qqnorm(dead2$age)
  qqline(dead2$age, lty = 1, col = "blue")
  t.test(dead2$age, mu = 28.92, alter = "two.side", conf.level = 0.95)
  
  #(3)단측 검정 - 방향성을 가짐
  t.test(dead2$age, mu = 28.92, alter= "greater", conf.level = 0.95)
  
  #(4)귀무가설의 임계값 계산
  qt(0.05,434,lower.tail = F)
  
  #귀무가설을 기각할 수 있는 임계값 = 1.64837
  #검정통계량 t=2.4023, 유의확률P=0.008355
  
  # 결론: 유의수준 0.05에서 귀무가설이 기각되므로,
  #       성별은 평균생존률에 차이가 있다.




### 상황2 ####

# 타이타닉호에 탑승한 승객들의 성별, 나이, 탑승위치등의 요소를 활용하여
# 생존확률을 예측한다.

#귀무가설: 탑승객의 데이터를 활용하여 생존확률을 예측할 수 없다.
#대립가설: 탑승객의 데이터를 활용하여 생존확률을 예측할 수 있다.

# 종속변수가 범주형이고 설명변수가 혼합형의 자료이므로,
# ML기반의 randomForest 기법이 활용 가능하다.

#### 1)랜덤포레스트 ####

  #(1)randomForest 패키지
  # 모델 생성
  titanic_RF_RF <- randomForest(survived~age+sex+embarked,
                                data=titanic_doBy_train,
                                na.action = na.omit,
                                ntree=100,
                                proximity=T)
  table(titanic_doBy_train$survived)
  plot(titanic_RF_RF,main="RF Model of titanic")
  importance(titanic_RF_RF) # 노드 불순도 개선에 기여한 변수: sex > age > embarked
  
  # 예측치 생성
  titan_pred_doBy<- predict(titanic_RF_RF,
                            newdata = titanic_doBy_test)
  confusionMatrix(titan_pred_doBy, factor(titanic_doBy_test$survived))
  
  # 정확도 약79%
  
  
  
  #(2)caret 패키지
  ctrl <- trainControl(method="cv", 3)
  titanic_RF_caret <- train(survived ~ age+sex+embarked,
                            data = titanic_caret_train,
                            na.action = na.omit,
                            method = 'cforest',
                            trControl = ctrl)
  titan_pred_caret<- predict(titanic_RF_caret,
                             newdata = titanic_caret_test)
  
  titan_caret_test <- subset(titanic_caret_test,!is.na(titanic_caret_test$age))
  confusionMatrix(titan_pred_caret, factor(titan_caret_test$survived))
  # 정확도 약77%




# iris 데이터 셋 ==============================================================

# 설명)
# Sepal.Length  continuous  꽃받침의 길이
# Sepal.Width   continuous  꽃받침의 폭
# Petal.Length  continuous  꽃잎의 길이
# Petal.width   continuous  꽃잎의 폭
# Species       factor      꽃의 종류


## 1.데이터 전처리 ============================================================
# 1)데이터 셋 불러오기
data3 <- iris

# 2)데이터 추출/자료형 변환
# 라벨링
iris_label <- ifelse(data3$Species == 'setosa', 0,
                     ifelse(data3$Species == 'versicolor', 1,2))
table(iris_label)
data3$label <- iris_label


sapply(data3,class)

# 3) train/test sets 생성
  #(1)doBy train/test sets 생성
  set.seed(1111)
  iris_doBy_train <- sampleBy(~Species, frac=0.7, data=data3) #전복의 성별을 기준으로 동일한 비율로 나눔
  iris_doBy_test <- sampleBy(~Species, frac=0.3, data=data3)
  
  iris_doBy_train_mat <- as.matrix(iris_doBy_train[-c(5:6)])
  iris_doBy_train_lab <- iris_doBy_train$label
  dim(iris_doBy_train_mat)
  length(iris_doBy_train_lab)
  
  #(2)caret train/test sets 생성
  set.seed(1000)
  iris_intrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE) 
  iris_caret_train <-iris[iris_intrain, ]
  iris_caret_test <-iris[-iris_intrain, ]
  table(iris_caret_train$Species)
  table(iris_caret_test$Species)




## 2.분석 =====================================================================
### 상황1 ####
# 여러가지 기법을 활용하여 붓꽃의 종류를 분류하고 가장 정확한 분류모델을 가려낸다.
#### 1)xgboost ####
  #xgBoost 패키지
  # 학습모델 생성
  iris_doBy_dtrain <- xgb.DMatrix(data = iris_doBy_train_mat,
                                  label = iris_doBy_train_lab)
  
  iris_doBy_xgb_model <- xgboost(data = iris_doBy_dtrain, max_depth = 2, eta = 1,
                                 nthread = 2, nrounds = 2,
                                 objective = "multi:softmax",
                                 num_class = 3,
                                 verbose = 0)
  iris_doBy_xgb_model
  
  # 모델 평가
  iris_doBy_test_mat <- as.matrix(iris_doBy_test[-c(5:6)])
  iris_doBy_test_lab <- iris_doBy_test$label
  
  doBy_pred_iris <- predict(iris_doBy_xgb_model, iris_doBy_test_mat)
  doBy_pred_iris
  
  table(doBy_pred_iris, iris_doBy_test_lab)
  
  (15+15+15) / length(iris_doBy_test_lab)
  # 정확도 100%
  
  # 주요변수 확인
  importance_matrix <- xgb.importance(colnames(iris_doBy_train_mat),
                                      model = iris_doBy_xgb_model)
  importance_matrix
  
  xgb.plot.importance(importance_matrix)
  
  
  
  #caret 패키지
  # 데이터 분할
  set.seed(123)
  idx = createDataPartition(data3$Species, list=F, p=0.7)
  Train = df[ idx,]
  Test  = df[-idx,]
  
  train.data  = as.matrix(Train[, names(data3)!="Species"])
  test.data   = as.matrix(Test[ , names(data3)!="Species"])
  train.label = as.integer(Train$Species) - 1 # 0기반
  test.label  = as.integer(Test$Species) - 1 # 0기반
  
  # 모델 생성
  dtrain = xgb.DMatrix(data=train.data, label=train.label)
  dtest  = xgb.DMatrix(data=test.data , label=test.label )
  watchlist = list(train=dtrain, eval=dtest)
  param = list(max_depth=2, eta=1, verbose=0, nthread=2,
               objective="multi:softprob", eval_metric="mlogloss", num_class=3)
  model = xgb.train(param, dtrain, nrounds=2, watchlist)
  
  # 테스트
  pred = as.data.frame(predict(model,test.data,reshape=T))
  names = levels(data3$Species)
  colnames(pred) = names
  pred$prediction = apply(pred,1,function(x) names[which.max(x)])
  pred$class = Test$Species
  pred
  table(pred$prediction, pred$class)
  
  #정분류율
  sum(pred$prediction==pred$class)/nrow(pred)
  #100%



#### 2)앙상블(배깅) ####
# adabag 패키지
  #(1)Bagging model 생성
  iris.bagging <- bagging(Species~., data=iris_doBy_train[1:5], mfinal=10)
  iris.bagging$importance
  
  #(2)도식화
  plot(iris.bagging$trees[[10]])
  text(iris.bagging$trees[[10]])
  
  #(3)예측값
  baggingpred <- predict(iris.bagging, newdata=iris)
  
  #(4)정오분류표
  baggingtb <- table(baggingpred$class, iris[,5])
  sum(baggingtb[row(baggingtb) == col(baggingtb)])/sum(baggingtb)  # 정분류율
  1-sum(baggingtb[row(baggingtb) == col(baggingtb)])/sum(baggingtb)  # 오분류율

# Caret Package
  #(1)Caret Package 배깅 모델 생성
  ctrl <- trainControl(method = 'cv',
                       number = 10)
  gyu <- train(Species ~ . ,
               data = iris_caret_train,
               method = 'treebag',
               trControl = ctrl)
  gyu
  
  #(2)예측 분류 결과 생성
  gyu_pred <- predict(gyu, newdata = iris_caret_test)
  
  #(3)적용 분류 결과 도출
  table(gyu_pred, iris_caret_test$Species)
  
  #(4)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(gyu_pred, iris_caret_test$Species)
  
  # 100% 신뢰수준을 확인 할 수 있다.


#### 3)앙상블(부스팅) ####
# adabag 패키지
  #(1)boosting model 생성
  boo.adabag <- boosting(Species~., data = iris_doBy_train,
                         boos = TRUE,
                         mfinal = 10)
  boo.adabag$importance
  
  #(2)도식화
  plot(boo.adabag$trees[[10]])
  text(boo.adabag$trees[[10]])
  
  #(3)예측값
  pred <- predict(boo.adabag, newdata = iris_doBy_test)
  
  #(4)정오분류표
  tb <- table(pred$class, iris_doBy_test[,5])
  sum(tb[row(tb) == col(tb)])/sum(tb)  # 정분류율
  1-sum(tb[row(tb) == col(tb)])/sum(tb)  # 오분류율
  
# Caret Package
  #(1)Caret Package 부스팅 학습 모델 설정
  ctrl <- trainControl(method = 'cv', number = 3) ## method : 샘플링을 하는 방법을 결정
  m1 <- train(Species ~ . , data = iris_caret_train,
                method = 'AdaBoost.M1',
                trControl = ctrl)
  m1
  
  #(2)예측 분류 결과 생성
  m1_pred <- predict(m1, newdata = iris_caret_test)
  
  #(3)적용 분류 결과 도출
  table(m1_pred, iris_caret_test$Species)
  
  #(4)모델 성능 평가 지표(정확도 확인)
  confusionMatrix(m1_pred, iris_caret_test$Species)
  # 100% 정확도를 확인할 수 있다.



  
# women 데이터 셋 ==============================================================

  # 설명)
  # height 여성의 키
  # weight 여성의 몸무게


## 1.데이터 전처리 ============================================================
# 1)데이터 셋 불러오기
data(women)
data5 <- women

# 2) train/test sets 생성
  #(1)doBy train/test sets 생성
  set.seed(1111)
  women_doBy_train <- sampleBy(~height, frac=0.7, data=data5) #전복의 성별을 기준으로 동일한 비율로 나눔
  women_doBy_test <- sampleBy(~height, frac=0.3, data=data5)
  
  #(2)caret train/test sets 생성
  set.seed(1000)
  women_intrain <- createDataPartition(y=data5$height, p=0.7, list=FALSE) 
  women_caret_train <-women[women_intrain, ]
  women_caret_test <- women[-women_intrain, ]




## 2.분석 =====================================================================
#### 1)단순선형회귀 ####
# stats 패키지
  #(1)모델 생성
  lm_women <- lm(weight ~ height, data = women_doBy_train)
  summary(lm_women)
  
  # 키가 몸무게에 미치는 영향
  # 위 결과 키가 몸무게에 대한 설명을 Adjusted R-squared:  0.9903 = 99%만큼 할 수 있다. 
  # p값 : 1.091e-14 / 0.05(유의수준) 작으므로 키는 몸무게에 영향을 미친다.
  
  # 선형회귀 시각화
  plot(women$weight, women$height)
  abline(lm_women, col = 'red')
  
  # 잔차 정규성 검정
  shapiro.test(lm_women$residuals)

# Caret Package
  #(1)모델 생성
  basic_model_lm <-train(weight ~.,
                         data=women_caret_train,
                         method="lm")
  summary(basic_model_lm) # 모델의 전체 요약값 
  
  regressControl  <- trainControl(method="repeatedcv", number = 4, repeats = 5) 
  
  regress <- train(weight ~ height,
                   data = women_caret_train,
                   method  = "lm",
                   trControl = regressControl, 
                   tuneGrid  = expand.grid(intercept = FALSE))
  
  summary(regress)



    
# economics 데이터 셋 ===========================================================

# 설명)
# data
# pce
# pop
# psavert
# uempmed
# unemploy


## 1.데이터 전처리 ============================================================
# 1)데이터 셋 불러오기
data("economics")
str(economics)




## 2.분석 =====================================================================
#### 1)시계열 분석 ####
# caret 패키지
plot(economics$unemploy,type = "l")
myTimeControl <- trainControl(method = "timeslice",
                              initialWindow = 36,
                              horizon = 12,
                              fixedWindow = TRUE)

plsFitTime <- train(unemploy ~ .,
                    data = economics,
                    method = "pls",
                    preProc = c("center", "scale"),
                    trControl = myTimeControl)
plsFitTime

# 3단계 : 예측
pred <- predict(plsFitTime, economics)

asdf <- cbind(pred, economics[,c(1,6)])
ggplot(asdf, aes(x = date, y = unemploy)) +
  geom_line(color = 'blue') + geom_line(aes(x = date, y = pred), color = 'red')





# pima.te 데이터 셋 ===========================================================

  # 설명)
  # height 여성의 키
  # weight 여성의 몸무게


## 1.데이터 전처리 ============================================================
# 1)데이터 셋 불러오기
data("Pima.te")
summary(Pima.te)   # 데이터의 구조 및 요약 정보를 살펴봅니다.




## 2.분석 =====================================================================
#### 1)ROC 커브 ####
# randomforest 패키지
#(1)모델 생성
Diag_DF <- data.frame(Attribute=c(colnames(Pima.te)[1:7]), AUC=NA)   # AUC 계산을 위한 데이터 프레임을 생성합니다. 

for(i in 1:nrow(Diag_DF)){
  roc_result <- roc(Pima.te$type, Pima.te[,as.character(Diag_DF$Attribute[i])])   # 확진 결과에 대한 데이터(type)와 진단 방법에 대한 후보 변수를 입력하여 AUC를 계산합니다. 
  Diag_DF[i,'AUC'] <- roc_result$auc}   # AUC 값을 입력합니다.

Diag_DF <- Diag_DF[order(-Diag_DF$AUC),]   # AUC 값을 오름차순 정렬합니다.
Diag_DF   # 결과를 확인해보면 "glu" 변수가 가장 좋은 성능임을 확인 할 수 있습니다.

# AUC가 가장 높은 "glu" 변수를 사용하여 ROC curve를 그리기
glu_roc <- roc(Pima.te$type, Pima.te$glu)   # "glu" 변수에 대한 ROC를 계산하여 value로 저장합니다.


plot.roc(glu_roc,   # roc를 계산한 value를 입력합니다.
         col="red",   # 선의 색상을 설정합니다.
         print.auc=TRUE,   # auc 값을 출력하도록 설정합니다.
         max.auc.polygon=TRUE,   # auc의 최대 면적을 출력하도록 설정합니다.
         print.thres=TRUE, print.thres.pch=19, print.thres.col = "red",   # 기준치(cut-off value)에 대한 출력, 포인트, 색상을 설정합니다.
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB")   # 선 아래 면적에 대한 출력, 색상을 설정합니다. 


# caret 패키지
#(1)모델 생성
a <- train(type ~ .,
           data = Pima.te,
           method='rocc')
plot(a)
summary(a)




