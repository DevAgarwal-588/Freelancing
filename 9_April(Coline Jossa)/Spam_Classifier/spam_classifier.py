# -*- coding: utf-8 -*-
"""Spam Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CRfA1sHAtjD2i_rpxqP36dDnjPnro1tz
"""

import numpy as np
acc = np.zeros((10,1))

"""# Multi-Nomial NB; CountVectorizer"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
mnb = MultinomialNB()
y_train=y_train.astype('int')
y_train

mnb.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=mnb.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[0] = count/1115.0

"""# Multi-Nomial NB; TfIDF"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
mnb = MultinomialNB()
y_train=y_train.astype('int')
y_train

mnb.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=mnb.predict(x_testcv)
a=np.array(y_test)
count=0
for i in range (len(predictions)):
    if (predictions[i]==a[i]):
      count=count+1
count
acc[1] = count/1115.0

"""# Logistic Regression; Count Vectorizer"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]
df.head()

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
lrc = LogisticRegression()
y_train=y_train.astype('int')
y_train

lrc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=lrc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[2] = count/1115.0

"""# Logistics Regression, TfIDF"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]
df.head()

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
lrc = LogisticRegression()
y_train=y_train.astype('int')
y_train

lrc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=lrc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[3] = count/1115.0

"""# SVM, Count Vectorizer"""

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
SVM = svm.SVC()
y_train=y_train.astype('int')
y_train

SVM.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=SVM.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[4] = count/1115.0

"""# SVM, TfIDF

"""

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
SVM = svm.SVC()
y_train=y_train.astype('int')
y_train

SVM.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=SVM.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[5] = count/1115.0

"""# RandomForestClassifier, Count Vectorizer"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
rfc = RandomForestClassifier(max_depth=2, random_state=0)
y_train=y_train.astype('int')
y_train

rfc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=rfc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[6] = count/1115.0

"""# RandomForestClassifier, TfIDF"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
rfc = RandomForestClassifier(max_depth=2, random_state=0)
y_train=y_train.astype('int')
y_train

rfc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=rfc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[7] = count/1115.0

"""# AdaBoost Classifier, Count Vectorizer"""

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
abc = AdaBoostClassifier(n_estimators=100, random_state=0)
y_train=y_train.astype('int')
y_train

abc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=abc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[8] = count/1115.0

"""# Ada Boost Classifier, TfIDF"""

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/content/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
abc = AdaBoostClassifier(n_estimators=100, random_state=0)
y_train=y_train.astype('int')
y_train

abc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=abc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[9] = count/1115.0

acc

result = np.where(acc == np.amax(acc))
result[0][0]

"""Q. Which methord is best to use for text based spam classification?

Multinomial Naive Bayes Classifier gave the highest accuracy with Count Vectorizer.

Naive Bayesian algorithm is a simple classification algorithm which uses probability of the events for its purpose. It is based on the Bayes Theorem which assumes that there is no interdependence amongst the variables.All of the properties contribute individually towards prediction and hence these features are referred to as “Naive”. As it considered the feature set to be Naive, the Naive Bayesian algorithm can be trained using less training data and also mislabeled data. 

Different kinds of Naive Bayesian implementations exist out of which one is Multinomial NB:
  It is generally used where there are discrete features(for example – word counts in a text classification problem). It generally works with the integer counts which are generated as frequency for each word. All features follow multinomial distribution. In such cases TF-IDF(Term Frequency, Inverse Document Frequency) also works.

The Bayesian statistics is different from the general statistics in various ways that a general probability calculation is always done around random events with a repeated number of trials while the Bayesian statistics is involved in calculating the prior and posterior probabilities. Bayesian statistics gives the leverage of the changing probabilities which can happen prior and post a certain event. 

hence, it is best methord for text based spam classification
"""

