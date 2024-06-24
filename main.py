###############################
#KNN
################################

import pandas as pd
from sklearn.metrics import  classification_report, roc_auc_score
from  sklearn.model_selection import  GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)


################################
#Exploratory Data Analysis
################################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

###########################################
#Data Preprocessing & Feature Engineering
###########################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis= 1)

#Descent Temlli yöntemlerde değişkenlerin standart olması
# elde edilecek sonuçların ya daha hızlı ya da daha doğru olmasını sağlayacaktır

X_scaled = StandardScaler().fit_transform(X)  # Bağımsız değişkenleri standartlaştırıyoruz

X= pd.DataFrame(X_scaled, columns = X.columns)

####################################
#Modeling & Prediction
####################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state= 47)

print(knn_model.predict(random_user)) # ben artık öğrendim tahmin edebilirim

##################################################
# Model Evaluate(Model Başarısı Değerlendirme)
##################################################

#ConfusionMatrix için y_pred
y_pred=knn_model.predict(X)

#AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1] #Bağımsız değişkenler ile 1 sınıfına ait olma olasılığını hesaplıyoruz

print(classification_report(y, y_pred))

#AUC
roc_auc_score(y, y_prob)
#0.90

#Holdout : sınama yöntemi
#Cross validation holdoutun bazi senaryolarda ortaya çıkarabileceği dezavantajları ortadan kaldırmak için kullanılır

cv_results = cross_validate(knn_model, X, y, cv = 5, scoring = ["accuracy", "f1", "roc_auc"])
print(cv_results)

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc']

#0.73
#0.59
#0.78

print(knn_model.get_params())

##################################################
# Hiperparametre Optimizasyonu (Hyperparameter)
##################################################

knn_model = KNeighborsClassifier()
knn_model.get_params() #Komşuluk sayısını 5 verdi

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best= GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs= -1,
                           verbose=1).fit(X, y) #Izgarada crfoss validation


knn_gs_best.best_params_

##############################################
#Final Model
##############################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y) #gcv den gelen parametreleri bu şekilde atayabiliriz

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


