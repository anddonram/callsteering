from call import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from keras.models import Model,Sequential,load_model
from keras.layers import Dense,Dropout, RepeatVector, Activation, LSTM, TimeDistributed, Input,recurrent,Masking
from keras.callbacks import ModelCheckpoint

import numpy as np
LSTM_LAYERS=1
DROPOUT=0.2
latent_dim=64
batch_size=32

def create_basic_model(model_filename,vocab_length,num_classes):
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of hidden_size
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    initialization = "he_normal"
    model.add(Dense(latent_dim,input_dim= vocab_length))
    model.add(Dropout(DROPOUT))
    model.add(Dense(num_classes, kernel_initializer=initialization))
    model.add(Dropout(DROPOUT))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_filename)

    return model
def create_seq2seq_model(model_filename):
    """Create the model"""
    input_token_index,target_token_index,\
    input_characters,target_characters,\
    max_encoder_seq_length,num_encoder_tokens,\
    max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

    initialization = "he_normal"
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of hidden_size
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    model.add(Masking(input_shape=(None, num_encoder_tokens)))
    for layer_number in range(LSTM_LAYERS):
        model.add(recurrent.LSTM(latent_dim,  kernel_initializer=initialization,
        return_sequences=layer_number + 1 < LSTM_LAYERS))
        model.add(Dropout(DROPOUT))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(max_decoder_seq_length))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(LSTM_LAYERS):
        model.add(recurrent.LSTM(latent_dim, return_sequences=True, kernel_initializer=initialization))
        model.add(Dropout(DROPOUT))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(num_classes, kernel_initializer=initialization)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_filename)

def train_model(model,model_filename,x_data,y_data,x_val,y_val,epochs):

    if epochs:
        checkpointer = ModelCheckpoint(filepath=model_filename,
        monitor='val_acc', verbose=0, save_best_only=True)
        model.fit(x_data, y_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val,y_val),
          callbacks=[checkpointer])


def scoring(y_target,y_pred,num_classes):
    conf_matrix=np.zeros((num_classes,num_classes))

    for expected,predicted in zip(y_target,y_pred):
        conf_matrix[expected,predicted]+=1

    tp=[conf_matrix[i,i] for i in range(conf_matrix.shape[0])]
    fp=np.sum(conf_matrix,axis=1)-tp
    fn=np.sum(conf_matrix,axis=0)-tp
    tn=sum(tp)-tp

    acc=sum(tp)/y_target.shape[0]
    prec=[(tp_)/(tp_+fp_) if (tp_+fp_)>0 else 0 for tp_,fp_,fn_,tn_ in zip(tp,fp,fn,tn)]
    rec=[(tp_)/(tp_+fn_) if (tp_+fn_)>0 else 0 for tp_,fp_,fn_,tn_ in zip(tp,fp,fn,tn)]
    return acc,np.mean(prec),np.mean(rec)

def validate_model(clf,pipe,num_classes):
    text,target,classes,classes_reverse=recover_from_files(eval_filepath)

    # Convert class to numeric
    y_test=[classes_reverse[x] for x in target]

    #Vectorize data
    x_test = pipe.transform(text)

    predict_test=model.predict(x_test)

    classes_test=[np.argmax(p) for p in predict_test]

    print("Final test")
    print(y_test)
    print(classes_test)
    acc,prec,rec= scoring(np.array(y_test),classes_test,num_classes)
    print(acc,prec,rec)

if __name__ =="__main__":
    text,target,classes,classes_reverse=recover_from_files(design_filepath)

    # Convert class to numeric
    y_data=[classes_reverse[x] for x in target]

    #Vectorize data
    count_vect = CountVectorizer(stop_words='english')
    tf_transformer = TfidfTransformer(use_idf=False)
    svd=TruncatedSVD(n_components=100)

    pipe=Pipeline([('counter',count_vect),
        ('tf_idf',tf_transformer),
        #('svd',svd)
        ])
    x_data = pipe.fit_transform(text)

    print("Número de textos",len(text))
    print("Tamaño del vocabulario",len(count_vect.vocabulary_))

    #Convert class to one-hot
    one_hot=OneHotEncoder()
    y_target=one_hot.fit_transform(np.array(y_data).reshape(-1,1))

    model_filename="model.h5"


    accuracy = []
    precision = []
    recall = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    x_data=x_data.toarray()

    create=False
    for train, val in kfold.split(x_data, y_data):

        x_train=x_data[train]
        y_train=y_target[train]
        x_val=x_data[val]
        y_val=y_target[val]

        model=None
        if create:
            model=create_basic_model(model_filename,len(count_vect.vocabulary_),len(classes))
            train_model(model,model_filename,x_train,y_train,x_val,y_val,10)
        else:
            model=load_model(model_filename)
        predict_val=model.predict(x_val)

        classes_val=[np.argmax(p) for p in predict_val]

        acc,prec,rec= scoring(np.array(y_data)[val],classes_val,len(classes))
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)


    print(np.mean(accuracy))
    print(np.mean(precision))
    print(np.mean(recall))

    validate_model(model,pipe,len(classes))
