# Demographic-prediction-using-NLP-in-url-keywords--MLP
#Assignment:  
When a user visits (one of ) our websites, we collect information about keywords extracted from the website's url. For each user, the frequency of visits per keyword per day is also stored. For example, suppose that a given user has visited the two following sites recently:  
html://mypage/abc-news/aaa-bbb.html   
html://mypage/news/aaa.html   
The keywords (that have been) “seen” by the user will be then stored as follows (semicolon is used to separate words):   abc:1;news:2;aaa:2;bbb:1;mypage:2   
Thanks to external data (or some sources of data bought by our marketing department), we have demographic information (like age, sex, race, ...) on about 5% of our visitors. The Head of Product wanted to predict demographics (age, sex) for the rest of our visitors from the keywords collected. He then spoke to Mr. Google, who advised him to hire a talented Master student from ESCP Europe, in order to transform his idea into reality. And now, you understand why we are contacting you ...   
The Head of Product's asked you to build a machine learning model to predict age and sex for each line in our dataset, which was partially extracted from one month's data (the portion of each day's data was concatenated). The dataset contains two files named train.csv (to help you train your model) and test.csv.   
Its format looks like: userID, keywords, age, sex (comma is used as a delimiter). Note that there are some missing data in our dataset, and we removed all the “labels” (age, sex) from the test file.

#For the Data preparation/cleaning process, we:  
1. dropped any row with a na value in keywords(both train and test).   
2. transferred the frenquency format(word:frequency;word:frequency...) to words format by frequency(word1 word1...word2 word2...).  
3. Cleaned the stopwords and applied stemporter and lowecase in text to normalize the words (we've thought about clean punctuations and pure numerical words but it shows lower accuracy in test).  
All the processes applied in both test and train file.
  
#For the model building:  
1. We applied TfidfVectorizer to change words to vectors, there are about 67k words in the final dictionary
2. For the "sex" prediction, we tried 3 classifiers: MultinomialNB, Logistic regression and Random Forest. based on the test result on AUC and accuracy, we chose Logistic regression(about 0.67 on AUC).  
3. For the "age" prediction, we tried 7 regressors: MultinomialNB, Logistic regression, Linear regression, Random Forest, Adabooster Regresso, Gradientboosting and SGD regressor. based on the test result on MSE, MAE, R2 and the accuracy of a age range of ±5 years(about 31% in the end), we chose SGD regressor(about 0.11 of R square and 10 of MAE).  

#For the Result file:  
  Please note that there are only 285 million rows in the result file cause there are about 26 millions rows of empty value(in keywords) in the original test file(311 million rows in total), we decide not to predict those empty rows.
