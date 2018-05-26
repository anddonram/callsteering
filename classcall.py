from call import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def simplify_classes(classes):
    res=[]
    res.extend(classes)

    clas_dict={}

    for clas in classes:
        for clas2 in classes:
            if clas2 in res:
                clas_dict[clas2]=clas2
                if clas!=clas2 and clas2.startswith(clas):
                    res.remove(clas2)
                    clas_dict[clas2]=clas

    generic_to_class={}
    for clas in clas_dict:
        generic_to_class.setdefault(clas_dict[clas],[]).append(clas)

    return res,clas_dict,generic_to_class

def rendimiento_final_combinado(clf,specific_classes,specific_classifiers,pipe):

    text,target,classes,classes_reverse=recover_from_files(eval_filepath)
    generic_classes,class_to_generic,generic_to_class=simplify_classes(classes)

    # Convert class to numeric
    y_test_original=[classes_reverse[x] for x in target]

    #Vectorize data
    x_test = pipe.transform(text)

    y_predicted=best.predict(x_test)
    for key in generic_to_class:
        specific_classes=generic_to_class[key]
        if(len(specific_classes)>1):
            #Multiclass
            #print("Predicting from generic class", key)
            indexes=np.array(y_predicted)==classes_reverse[key]

            x_bank=x_test[indexes]
            sp_best=specific_classifiers[classes_reverse[key]]

            y_predicted[indexes]=sp_best.predict(x_bank)

    print("Test Combined accuracy",sum(y_pred==y_real for y_pred,y_real in zip(y_predicted,y_test_original))/len(y_test_original))

if __name__=="__main__":
    # generic_classes=["BankAccount",
    # "BankOfficeOpeningInfo",
    # "CreditCard",
    # "Login",
    # "MoneyTransfer",
    # "NearestATM",
    # "NearestOffice",
    # "TransferAgent"]


    text,target,classes,classes_reverse=recover_from_files(design_filepath)

    generic_classes,class_to_generic,generic_to_class=simplify_classes(classes)
    print(generic_classes)
    # Convert class to numeric
    y_original=np.array([classes_reverse[x] for x in target])
    y_data=[classes_reverse[class_to_generic[x]] for x in target]

    #Vectorize data
    count_vect = StemmedCountVectorizer(stop_words='english')
    #use_idf makes classification worse
    tf_transformer = TfidfTransformer(use_idf=False)

    #svd_components=100
    #svd=TruncatedSVD(n_components=svd_components)
    #scaler=MinMaxScaler()
    pipe=Pipeline([('counter',count_vect),
        ('tf_idf',tf_transformer),
    #    ('svd',svd),
        #('scaler',scaler)
        ])
    x_data = pipe.fit_transform(text)

    print("Número de textos",len(text))
    print("Tamaño del vocabulario",len(count_vect.vocabulary_))


    # parameters =  {'alpha': [0.01, 1, 5]}
    # clf=MultinomialNB()
    # best=busqueda_cv(clf, x_data,y_data,parameters)

    # parameters =  {'penalty':('l1','l2'),'C': [0.1, 1, 10]}
    # clf = LogisticRegression()
    # best=busqueda_cv(clf, x_data,y_data,parameters)

    parameters = [
      {'C': [0.1, 1, 10], 'kernel': ['linear','poly']},
      {'C': [0.1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    clf = SVC()
    best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters={'min_samples_split':[0.0001,0.001,0.01]}
    # clf=DecisionTreeClassifier()
    # best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters={'max_iter': [1000] ,'tol':[1e-3],
    #     'loss':('hinge','log'), 'alpha':[0.0001,0.001]}
    # clf=SGDClassifier()
    # best=busqueda_cv(clf, x_data,y_data,parameters)
    #
    # parameters={'n_estimators':[5,10,15],
    # 'criterion':('gini','entropy'),
    # 'max_features':[0.2,0.5,'auto'],
    # 'min_samples_split':[0.0001,0.001,0.01,0.1],
    # 'min_samples_leaf':[0.001,0.01,0.1]}
    # clf= RandomForestClassifier()
    # best=busqueda_cv(clf, x_data,y_data,parameters)

    # predictions=best.predict(x_data)
    # for y_pred,y_target,txt in zip(predictions,y_data,text):
    #     if(y_pred!=y_target):
    #         print(txt,"Expected",classes[y_target],"Predicted",classes[y_pred])


    specific_classifiers={}
    for key in generic_to_class:
        specific_classes=generic_to_class[key]
        if(len(specific_classes)>1):
            #Multiclass
            print("Classifying from generic class", key)
            indexes=np.array(y_data)==classes_reverse[key]

            x_bank=x_data[indexes]
            y_bank=y_original[indexes]

            parameters = [
              {'C': [0.1, 1, 10], 'kernel': ['linear','poly']},
              {'C': [0.1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
             ]
            clf = SVC()
            sp_best=busqueda_cv(clf, x_bank,y_bank,parameters)
            specific_classifiers[classes_reverse[key]]=sp_best
        else:
            specific_classifiers[classes_reverse[key]]=None


    y_predicted=best.predict(x_data)
    for key in generic_to_class:
        specific_classes=generic_to_class[key]
        if(len(specific_classes)>1):
            #Multiclass
            print("Predicting from generic class", key)
            indexes=np.array(y_predicted)==classes_reverse[key]

            x_bank=x_data[indexes]
            sp_best=specific_classifiers[classes_reverse[key]]

            y_predicted[indexes]=sp_best.predict(x_bank)
    print("Combined accuracy",sum(y_pred==y_real for y_pred,y_real in zip(y_predicted,y_original))/len(y_original))
    #rendimiento_final_combinado(best,specific_classes,specific_classifiers,pipe)
