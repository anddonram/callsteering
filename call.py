import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import make_scorer,precision_score,recall_score,get_scorer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
design_filepath="CorpusCallSteering/Design"
eval_filepath="CorpusCallSteering/Evaluation"

scoring={'acc':'accuracy',
    'prec':make_scorer(precision_score,average="micro"),
    'rec':make_scorer(recall_score,average="micro"),}


def recover_from_files(filepath):
    files=sorted(os.listdir(filepath))

    classes=[]
    classes_reverse={}
    text=[]
    target=[]
    for f in files:
        with open(os.path.join(filepath,f)) as file:
            classification=f[:f.find(".")]

            classes_reverse[classification]=len(classes)
            classes.append(classification)

            for line in file:
                text.append(line)
                target.append(classification)

    # print(text)
    return text,target,classes,classes_reverse


def ver_entrenamiento(clf,x_data,y_data,cv=5):

    print(clf)
    cross_val=cross_validate(clf, x_data,y_data,
            cv=cv,scoring=scoring,return_train_score=True)
    keys=sorted(cross_val)
    for k in keys:
        print(k,cross_val[k].mean())

def busqueda_cv(clf, x_data,y_data,parameters,cv=5,metric='acc'):

    grid = GridSearchCV(clf, parameters,scoring=scoring,cv=cv,
                refit=metric).fit(x_data,y_data)
    best=grid.best_estimator_
    print(best)
    print(grid.best_score_)
    return best

def rendimiento_final(clf,x_data,y_data,count_vect):

    text,target,classes,classes_reverse=recover_from_files(eval_filepath)

    # Convert class to numeric
    y_test=[classes_reverse[x] for x in target]

    #Vectorize data
    x_test = count_vect.transform(text)
    print("Final test")
    for key in scoring:
        print(key,get_scorer(scoring[key])(clf,x_test,y_test))

if __name__=="__main__":
    text,target,classes,classes_reverse=recover_from_files(design_filepath)

    # Convert class to numeric
    y_data=[classes_reverse[x] for x in target]

    # text_train, text_val, y_train, y_val=train_test_split(text,y_data,random_seed=30,
    #     test_size=0.2)

    #Vectorize data
    count_vect = CountVectorizer(stop_words='english')
    x_data = count_vect.fit_transform(text)

    print("Número de textos",len(text))
    print("Tamaño del vocabulario",len(count_vect.vocabulary_))

    parameters =  {'alpha': [0.01, 1, 5]}
    clf=MultinomialNB()
    best=busqueda_cv(clf, x_data,y_data,parameters)

    # parameters =  {'penalty':('l1','l2'),'C': [0.1, 1, 10]}
    # clf = LogisticRegression()
    # best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters = [
    #   {'C': [0.1, 1, 10], 'kernel': ['linear']},
    #   {'C': [0.1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    #  ]
    # clf = SVC()
    # best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters={'min_samples_split':[0.0001,0.001,0.01]}
    # clf=DecisionTreeClassifier()
    # best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters={'loss':('hinge','log'), 'alpha':[0.0001,0.001]}
    # clf=SGDClassifier()
    # best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters={'n_estimators':[5,10], 'criterion':('gini','entropy'),'min_samples_split':[0.0001,0.001,0.01]}
    # clf= RandomForestClassifier()
    # best=busqueda_cv(clf, x_data,y_data,parameters)

    predictions=best.predict(x_data)
    for y_pred,y_target,txt in zip(predictions,y_data,text):
        if(y_pred!=y_target):
            print(txt,"Expected",classes[y_target],"Predicted",classes[y_pred])

    #rendimiento_final(best,x_data,y_data,count_vect)
