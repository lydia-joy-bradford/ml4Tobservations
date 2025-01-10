# # Setup
## Load packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import scipy
#import important sklearn packages 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
# import time for calculating the time it takes to train
import time as time
#visualization packages 
from yellowbrick.target import ClassBalance
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.style.palettes import PALETTES, SEQUENCES, color_palette
from yellowbrick.style import set_palette
# RNN packages
import tensorflow as tf
from tensorflow import keras

# Plot and save the figures
# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Load data
'''
This data comes from a clean dataset. When cleaning the dataset, make sure that the data is in long format with the following variables: "score", "rater", "text", "question" with no missing text data or score data (once the automatic scoring is finetuned, it can be used to predict the score data). If there is missing rater or question data, these can be imputed with minimal effects on the machine learning. 
'''
df = pd.read_excel('obs-scores.xlsx', index=False)

# Exploratory data analysis 
'''
Here, take a quick look at the distribution of the raters and questions
'''
# Get the distributoin of the scores
new_df = df['score'].value_counts(normalize=True)
new_df = new_df.mul(100).rename('Percent').reset_index()
new_df = new_df.rename(columns={'index':'scores'})

# set palette for looking at the graph of the distribution of the scores
palette = sns.light_palette("seagreen")

# catplot of the distribution
g = sns.catplot(x='scores', y='Percent', kind='bar', palette=palette, data=new_df)
g.ax.set_ylim(0,100)
save_fig('Score Distribution')
#Quick statistics on the raters and questions 
df[['rater','Question']].describe()

# Get the destribution of how many scores each rater gave
df[['rater']].value_counts()
#See how many different questions there are 
df[['Question']].value_counts()


### Baseline
'''
Here, the baseline of randomly assigned scores if built based upon the distribution found above. With each score being assigned with the probability of what they were assigned in the actual dataset. 
'''
np.random.seed(123)
df['predict'] = np.random.choice(np.arange(1,5),4925,p=[0.07,0.20,0.34,0.39])

# report on baseline
'''
here,the precision, recall, and f1-scores are reported on the baseline as well as the confusion matrix which shows where different scores are correctly or incorrectly predicted.
'''
df['predict'].describe() #  Check statistics of the "predicted" value

print(classification_report(df['predict'],df['score']))
cf_matrix = confusion_matrix(df['score'],df['predict'])
print(cf_matrix)

#heatmap of the confusion matrix 
label = [1,2,3,4]
hm = sns.heatmap(cf_matrix, annot=True, cbar=False, xticklabels = label, yticklabels = label, fmt = 'g',cmap='Greens')
save_fig('Baseline-confusion-matrix')

# Training and Cross Validation
'''
Here, the data is prepared for the training and cross validation and is then trained and cross validated across logistic regression, Bernoulli Naive Bayes, Decision Tree, Random Forest, and RNN. 

The Cross validation starts with a randomized cross validation and then a grid search near the optimal parameters from the randomized cross validation. These were done for logistic regression-random forest. 

Only a couple of parameters were compared across the RNN. 
'''

#vectorize text
vectorizer = CountVectorizer()
X_matrix = vectorizer.fit_transform(df[['text']].to_numpy().ravel())

#split training and testing set for vectorized data
X_train, X_test, y_train,y_test = train_test_split(X_matrix,
        df[['score']].to_numpy().ravel(),
        test_size=.2,random_state=123)

# split training and testing for full dataframe (this is for the RNN)
df_train, df_test = train_test_split(df,
        test_size=.2,random_state=123)

#removing very short text (these do not work well with the RNN)
df_train = df_train[df_train['text'].str.len()>5]
df_test = df_test[df_test['text'].str.len()>5]


# # Logistic Regression

# random search for logistic regression
lr_param_grid = [
    {'penalty':['none','l2','l1','elasticnet'],'C':uniform(loc=0,scale=4),
    'solver':['newton-cg','sag','saga','lbfgs']}
]

lr = LogisticRegression(max_iter = 1000)
lr_random_search = RandomizedSearchCV(lr, lr_param_grid,n_iter = 100, random_state=123)
lr_random_search.fit(X_train, y_train)

# best parameters from randomized search
lr_random_search.best_params_

# Grid Search for logistic regression - 5 CVs 
glr = LogisticRegression(max_iter = 1000,penalty = 'l2', solver= 'lbfgs')
glr_gridsearch = GridSearchCV(glr,param_grid = {'C':np.arange(0.25,0.35,0.0001)})
glr_gridsearch.fit(X_train, y_train)

#optimal parameters
glr_gridsearch.best_params_

#define the Optimal Logistic Regression
olr = LogisticRegression(max_iter = 1000,penalty = 'l2', solver= 'lbfgs',C=0.303)

#time for training of the ML method
time_dic = {'type':[],'time':[]}
time_dic['type'].append('Logistic')

# fit the optimal logistic regression
start = time.time()
olr.fit(X_train,y_train)
end = time.time()
time_dic['time'].append(end-start)

#predict
prediction = olr.predict(X_test)

#Confusion matrix for the optimal logistic regression
confusion_matrix(prediction,y_test)

#classification report for the optimal logistic regression
print(classification_report(y_test,prediction))


# # Bernoulli Naive Bayes Classifier

# Parameters for Randomized Search CV
NB_param_grid = [
    {'alpha':uniform(0,4)}
]
nb = BernoulliNB()

nb_random_search = RandomizedSearchCV(nb,NB_param_grid,n_iter = 100,
                                      random_state=123)
nb_random_search.fit(X_train,y_train)

# random cv best parameters
nb_random_search.best_params_

# Grid Search CV
nb_gridsearch = GridSearchCV(nb,param_grid={'alpha':np.arange(0.4,0.5,0.0001)})
nb_gridsearch.fit(X_train,y_train)

# optimal parameters
nb_gridsearch.best_params_

# define the BNB model with the optimal parameters
onb = BernoulliNB(alpha=0.4334)

#add time to train variable for Naive Bayes 
time_dic['type'].append('bernoulliNB')

#fit the BNB with the optimal parameters
start = time.time()
onb.fit(X_train,y_train)
end = time.time()
print(end-start)
time_dic['time'].append(end-start)

#predict on the testing set
pred_onb = onb.predict(X_test)

#classification report for the optimal BNB
print(classification_report(y_test,pred_onb))

# # Decision Tree

# define the parameters for the randomized search
DTC_param_grid = [
    {'criterion':['gini','entropy'],'splitter':['best','random'],
    'min_samples_split':randint(1,40),'min_samples_leaf':randint(1,20)
    }
]
dtc = DecisionTreeClassifier()
#randomized search for the optimal hyperparameters
dtc_random_search = RandomizedSearchCV(dtc,DTC_param_grid,n_iter = 100,
                                      random_state=123)
dtc_random_search.fit(X_train,y_train)

#randomized cv best parameters
dtc_random_search.best_params_

#Grid search
grid_param = {'min_samples_split':np.arange(30,41,1),'min_samples_leaf':np.arange(1,10,1)}
dtc = DecisionTreeClassifier(criterion = 'entropy',splitter = 'random')
dtc_gridsearch = GridSearchCV(dtc,grid_param)
dtc_gridsearch.fit(X_train,y_train)
 
#optimal parameters
dtc_gridsearch.best_params_

#Fit decision tree with optimal parameters 
odtc = DecisionTreeClassifier(criterion = 'entropy',splitter = 'random',min_samples_leaf = 1,min_samples_split = 33)

#Add training time for the decision tree 
time_dic['type'].append('DTC')

start = time.time()
odtc.fit(X_train,y_train)
end = time.time()
print(end-start)
time_dic['time'].append(end-start)

#predict scores from the testing set with the trained decision tree
odtc_pred = odtc.predict(X_test)

#classification report
print(classification_report(y_test,odtc_pred))


# # Random Forest
# set the grid for the randomized search
RFC_param_grid = [
    {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
]
rfc = RandomForestClassifier()

#Random search for optimal hyperparameters 
rfc_random_search = RandomizedSearchCV(rfc,RFC_param_grid,n_iter = 100,
                                      random_state=123)
rfc_random_search.fit(X_train,y_train)

#randomized search bets parameters 
rfc_random_search.best_params_

#set parameters for those not explored in the grid. 
rfc = RandomForestClassifier(min_samples_leaf = 1,max_features = 'auto',bootstrap=False)

#grid search
grid_param = {'min_samples_split':[1,2,3,4],'n_estimators':np.arange(1300,1500,10),'max_depth':np.arange(55,66,1)}
rfc_gridsearch = GridSearchCV(rfc,grid_param,verbose=3)
rfc_gridsearch.fit(X_train,y_train)

#optimal parameters 
rfc_gridsearch.best_params_

#define the optimal random forest classifier
orfc = RandomForestClassifier(bootstrap=False, max_depth=65, min_samples_split=2,
                       n_estimators=1800)
#add time to train for Random Forest 
time_dic['type'].append('RFC')

#fit the optimal random forest 
start = time.time()
orfc.fit(X_train,y_train)
end = time.time()
print(end-start)
#time_dic['time'].append(end-start)

#predict scores from the testing set with the random forest
orfc_pred = orfc.predict(X_test)

#classification report
print(classification_report(y_test,orfc_pred))


# # RNN

'''
This is adapted/taken from: 
https://www.tensorflow.org/text/tutorials/text_classification_rnn

'''

#establish the training and testing datasets within a tensor data set
text = ['text']
score = ['score']
train_ds = (tf.data.Dataset.from_tensor_slices(
        (
        tf.cast(df_train[text].values,tf.string),
        tf.cast(df_train[score].values,tf.int64)
        )
    )
)
test_ds = (tf.data.Dataset.from_tensor_slices(
        (
        tf.cast(df_test[text].values,tf.string),
        tf.cast(df_test[score].values,tf.int64)
        )
    )
)


#encoder for the RNN
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_ds.map(lambda text, label: text))

#Check whether encoder works 
vocab = np.array(encoder.get_vocabulary())
vocab[:20]
encoded_example = encoder(df_train[['text']])[:3].numpy()
encoded_example

for n in range(3):
    print("Original:", df_train[['text']].iloc[n])
    print("Round-trip:"," ".join(vocab[encoded_example[n]]))
    print()
    
#first rnn model
model1 = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim = 64, 
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(5)
    ])

#compile the rnn model
model1.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             optimizer=tf.keras.optimizers.Adam(1e-4),
             metrics=['accuracy'])

#fit the model with the training data, validate with testing
history = model1.fit(train_ds,epochs=10,
                   validation_data = test_ds,
                   validation_steps=30)

#make predictions on the testing data
predictions = model1.predict(test_ds)
#define the predicted classes
classes = np.argmax(predictions, axis = 1)

#classification report
print(classification_report(df_test['score'],classes))

# 2nd RNN options -- add an additional bidirectional layer and a dropout option
model2 = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5)
])

#compile 2nd model
model2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#fit second model
history = model2.fit(train_ds, epochs=10,
                    validation_data=test_ds,
                    validation_steps=30)

predictions = model2.predict(test_ds)
classes = np.argmax(predictions, axis = 1)
print(classification_report(df_test['score'],classes))

# # Include Observer for observer bias
df['rater']=df['rater'].astype(str)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
rater = df['rater'].to_numpy().ravel()
rater = rater.reshape(-1,1)
r_matrix = enc.fit_transform(rater)

# # Training
#vectorize text
vectorizer = CountVectorizer()
X_matrix2 = vectorizer.fit_transform(df2[['text']].to_numpy().ravel())
r_matrix = df2['rater'].to_numpy().ravel()
r_matrix = np.reshape(r_matrix,(4925,1))
X_matrix3 = scipy.sparse.hstack((r_matrix,X_matrix2))

#split training and testing set for vectorized data
X_train, X_test, y_train,y_test = train_test_split(X_matrix3,
        df2[['score']].to_numpy().ravel(),
        test_size=.2,random_state=123)

# ## Logistic Regression
start = time.time()
olr.fit(X_train,y_train)
end = time.time()
print(end-start)

#predict
prediction = olr.predict(X_test)

#classification report for the optimal logistic regression
print(classification_report(y_test,prediction))

# ## Bernoulli Naive Bayes Classifier

onb = BernoulliNB(alpha=0.4334)

#BNB with the optimal parameters
start = time.time()
onb.fit(X_train,y_train)
end = time.time()
print(end-start)
pred_onb = onb.predict(X_test)

#classification report for the optimal BNB
print(classification_report(y_test,pred_onb))


# ## Decision Tree 
odtc = DecisionTreeClassifier(criterion = 'entropy',splitter = 'random',min_samples_leaf = 1,min_samples_split = 33)
start = time.time()
odtc.fit(X_train,y_train)
end = time.time()
print(end-start)

#predict scores from the testing set with the trained decision tree
odtc_pred = odtc.predict(X_test)

#classification report
print(classification_report(y_test,odtc_pred))


# ## Random Forest

#the optimal random forest classifier
orfc = RandomForestClassifier(bootstrap=False, max_depth=65, min_samples_split=2,
                       n_estimators=1800)

#fit the optimal random forest 
start = time.time()
orfc.fit(X_train,y_train)
end = time.time()
print(end-start)

#predict scores from the testing set with the random forest
orfc_pred = orfc.predict(X_test)

#classification report
print(classification_report(y_test,orfc_pred))


# # With Question Label
question = df['Question'].to_numpy().ravel()
question = question.reshape(-1,1)
q_matrix = enc.fit_transform(question)
X_matrix3 = scipy.sparse.hstack((q_matrix,X_matrix2))

#split training and testing set for vectorized data
X_train, X_test, y_train,y_test = train_test_split(X_matrix3,
        df3[['score']].to_numpy().ravel(),
        test_size=.2,random_state=123)


# ## Logistic Regression
start = time.time()
olr.fit(X_train,y_train)
end = time.time()
print(end-start)
#predict
prediction = olr.predict(X_test)
#classification report for the optimal logistic regression
print(classification_report(y_test,prediction))

# ## Bernoulli Naive Bayes Classifier

#BNB with the optimal parameters
start = time.time()
onb.fit(X_train,y_train)
end = time.time()
print(end-start)
pred_onb = onb.predict(X_test)
#classification report for the optimal BNB
print(classification_report(y_test,pred_onb))

# ## Decision Tree
start = time.time()
odtc.fit(X_train,y_train)
end = time.time()
print(end-start)

#predict scores from the testing set with the trained decision tree
odtc_pred = odtc.predict(X_test)

#classification report
print(classification_report(y_test,odtc_pred))


# ## Random Forest

#optimal random forest 
start = time.time()
orfc.fit(X_train,y_train)
end = time.time()
print(end-start)

#predict scores from the testing set with the random forest
orfc_pred = orfc.predict(X_test)

#classification report
print(classification_report(y_test,orfc_pred))


# ## With Both
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
df['rater']=df['rater'].astype(str)
rater = df['rater'].to_numpy().ravel()
rater = rater.reshape(-1,1)
r_matrix = enc.fit_transform(rater)
question = df['Question'].to_numpy().ravel()
question = question.reshape(-1,1)
q_matrix = enc.fit_transform(question)
vectorizer = CountVectorizer()
X_matrix2 = vectorizer.fit_transform(df4[['text']].to_numpy().ravel())
X_matrix3 = scipy.sparse.hstack((q_matrix,X_matrix2))
X_matrix4 = scipy.sparse.hstack((r_matrix,X_matrix3))

#split training and testing set for vectorized data
X_train, X_test, y_train,y_test = train_test_split(X_matrix4,
        df4[['score']].to_numpy().ravel(),
        test_size=.2,random_state=123)


# ### Logistic Regression
# Optimal Logistic Regression
olr = LogisticRegression(max_iter = 1000,penalty = 'l2', solver= 'lbfgs',C=0.303)
start = time.time()
olr.fit(X_train,y_train)
end = time.time()
print(end-start)
cm = ConfusionMatrix(olr,cmap='greens')
cm.fit(X_train,y_train)
cm.score(X_test,y_test)
save_fig('confusion-matrix1')
cm.show()
#predict
prediction = olr.predict(X_test)
#classification report for the optimal logistic regression
print(classification_report(y_test,prediction))

# ### Bernoulli Naive Bayes

# BNB model with the optimal parameters
onb = BernoulliNB(alpha=0.4334)

#fit the BNB with the optimal parameters
start = time.time()
onb.fit(X_train,y_train)
end = time.time()
print(end-start)
pred_onb = onb.predict(X_test)
#classification report for the optimal BNB
print(classification_report(y_test,pred_onb))



