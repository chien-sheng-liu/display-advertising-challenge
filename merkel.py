import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import ADASYN
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss, roc_auc_score, classification_report, f1_score, precision_score, recall_score
import mlflow
import mlflow.keras
from deepctr.layers.utils import Concat
from deepctr.layers import Concat
from deepctr.layers import Linear
from deepctr.layers import DNN, NoMask

class Preprocessing:
    @staticmethod
    def feature_engineering(df):
        numerical_features = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        categorical_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                                'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

        df[categorical_features] = df[categorical_features].astype(str)

        # Label encode categorical features
        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])

        X_numerical = df[numerical_features].values
        X_categorical = df[categorical_features].values
        y = df['label'].values

        return X_numerical, X_categorical, y
    
    @staticmethod
    def deepfm_pre_setting(df):
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]

        # Define a dictionary to store maximum encoded values
        max_encoded_values = {}

        # Find maximum encoded values for each categorical feature
        for feat in sparse_features:
            max_encoded_values[feat] = df[feat].max()  # Assuming numerical encoding
        
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=max_encoded_values[feat] + 1, embedding_dim=8)
                       for feat in sparse_features] + [DenseFeat(feat, 1,)
                        for feat in dense_features]
                       
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        
        return feature_names, dnn_feature_columns, linear_feature_columns, max_encoded_values
    
    @staticmethod
    def splitting_dataset(df, feature_names):
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        test_size = len(df) - train_size - val_size

        train_df = df[:train_size]
        val_df = df[train_size:train_size+val_size]
        test_df = df[train_size+val_size:]

        train_model_input = {name: train_df[name] for name in feature_names}
        val_model_input = {name: val_df[name] for name in feature_names}
        test_model_input = {name: test_df[name] for name in feature_names}
        
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values

        return train_model_input, val_model_input, test_model_input, y_train, y_val, y_test
    
    @staticmethod
    def resampling(feature_names, train_model_input, y_train):
        adasyn = ADASYN(sampling_strategy='auto', random_state=42)
        train_model_input_df = pd.DataFrame(train_model_input)
        train_model_input_resampled, y_train_resampled = adasyn.fit_resample(train_model_input_df, y_train)
        train_model_input_resampled = {name: train_model_input_resampled[name].values for name in feature_names}
        
        return train_model_input_resampled, y_train_resampled

class DeepFMTraining:
    @staticmethod
    def train_model(linear_feature_columns, dnn_feature_columns, train_model_input, y_train, val_model_input, y_val, 
                    batch_size, epochs, dnn_hidden_units, dnn_dropout, l2_reg_dnn, patience=10):
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', 
                       dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout, l2_reg_dnn=l2_reg_dnn)
        model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=['Precision', 'Recall', 'accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = model.fit(train_model_input, y_train, batch_size=batch_size, epochs=epochs, verbose=2, 
                            validation_data=(val_model_input, y_val), callbacks=[early_stopping])
        
        return history, model
    
    @staticmethod
    def test_model(model, batch_size, history, test_model_input, y_test):
        pred_ans = model.predict(test_model_input, batch_size=batch_size)
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in pred_ans]
        
        logloss = log_loss(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        class_f1 = classification_report(y_test, y_pred_binary, output_dict=True)
        
        print("Test LogLoss:", round(logloss, 4))
        print("Test AUC:", round(auc, 4))
        print("Classification Report:")
        print(pd.DataFrame(class_f1))
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.savefig('learning_curve.png')
        
        return y_pred_binary

if __name__ == '__main__':
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: The file 'train.csv' was not found.")
        exit()

    preprocessing = Preprocessing()
    X_numerical, X_categorical, y = preprocessing.feature_engineering(df)
    feature_names, dnn_feature_columns, linear_feature_columns, max_encoded_values = preprocessing.deepfm_pre_setting(df)
    train_model_input, val_model_input, test_model_input, y_train, y_val, y_test = preprocessing.splitting_dataset(df, feature_names)
    #train_model_input, y_train = preprocessing.resampling(feature_names, train_model_input, y_train)
    
    # 模型訓練參數設定
    batch_size = 512
    epochs = 300
    dnn_hidden_units = [512, 256, 256]
    dnn_dropout = 0.3
    l2_reg_dnn = 1e-04
    patience = 10
        
    mlflow.set_experiment("DeepFM Experiment")
    with mlflow.start_run():
        try:
            history, model = DeepFMTraining.train_model(linear_feature_columns, dnn_feature_columns, train_model_input, y_train, 
                                                 val_model_input, y_val, batch_size=batch_size, epochs=epochs, 
                                                 dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout, 
                                                 l2_reg_dnn=l2_reg_dnn, patience=patience)

            y_pred_binary = DeepFMTraining.test_model(model, batch_size, history, test_model_input, y_test)

            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("dnn_hidden_units", dnn_hidden_units)
            mlflow.log_param("dnn_dropout", dnn_dropout)
            mlflow.log_param("l2_reg_dnn", l2_reg_dnn)
            mlflow.log_param("patience", patience)

        except Exception as e:
            print("Error during training and evaluation:", e)
            mlflow.end_run()  # 確保在出錯時也結束 MLflow 的 run
            exit()

    mlflow.end_run()  # 確保 MLflow 正常結束

    # 繼續執行後續程式碼
    df_predict = pd.read_csv("test.txt", sep="\t")
    print(df_predict.head())

    col = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    df_predict.columns = col
    
    numerical_features = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
    categorical_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    
    # 填補數值資料的空缺
    df_predict[numerical_features] = df_predict[numerical_features].fillna(df_predict[numerical_features].mean())
    
    # 填補類別資料的空缺
    df_predict[categorical_features] = df_predict[categorical_features].fillna(df_predict[categorical_features].mode().iloc[0])

    # 標準化數值型欄位
    scaler = StandardScaler()
    df_predict[numerical_features] = scaler.fit_transform(df_predict[numerical_features])
    
    # 檢查和處理超過最大編碼值的情況
    for col in numerical_features:
        max_val = df[col].max()
        df_predict[col] = df_predict[col].apply(lambda x: max_val if x > max_val else x)

    # 轉換類別型欄位為字符串
    df_predict[categorical_features] = df_predict[categorical_features].astype(str)

    print(df_predict.head(5))

    # Label encode categorical features
    for col in categorical_features:
        df_predict[col] = LabelEncoder().fit_transform(df_predict[col])
        
    for col in categorical_features:
        max_val = max_encoded_values[col]
        df_predict[col] = df_predict[col].apply(lambda x: max_val if x > max_val else x)

    prediction_input = {name: df_predict[name].values for name in feature_names} 

    # 批次處理預測
    batch_size = batch_size
    num_batches = len(df_predict) // batch_size + 1
    all_predictions = []

    for i in range(num_batches):
        batch_data = {name: values[i * batch_size: (i + 1) * batch_size] for name, values in prediction_input.items()}
        batch_predictions = model.predict(batch_data, batch_size=batch_size)
        all_predictions.extend(batch_predictions.flatten())

    start_id = 60000000
    id_list = list(range(start_id, start_id + len(all_predictions)))
    
    df_predictions = pd.DataFrame({
        'ID': id_list,
        'Predicted': all_predictions
    })

    df_predictions.to_csv('prediction_result.csv', index=False)

    print("Predictions saved to 'prediction_result.csv'.")