# Importing the required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from Functions import preprocess
from Functions import EncodingandSplit
from sklearn.metrics import roc_curve,confusion_matrix,accuracy_score,precision_score,f1_score,classification_report,recall_score,roc_auc_score,log_loss,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score,KFold
class classificationModel():
    def RF(self,rawData):
        self.rawData=rawData
        ppobj=preprocess.dataPreprocess()
        edsobj=EncodingandSplit.encodingandsplit()
        preProcessedData=ppobj.preProcessing(self.rawData)
        print(preProcessedData.columns)
        #features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU','Final Group', 'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the query ', 'Lastest RAG Status']
        #features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU','FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query', 'Lastest RAG Status','Reason for the RAG Status ']
        features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU', 'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the RAG Status ']
        df_rf_le = preProcessedData[features]
        df_rf_le_y = preProcessedData['Joining Status']
        print(df_rf_le_y.value_counts())
        df_rf_le=edsobj.labelEncoding(df_rf_le)
        X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf=edsobj.traintestsplit(df_rf_le,df_rf_le_y,100)
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train_rf_clf,y_train_rf_clf)
        return rf_clf,X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf
    def preProcesstestData(self,rawData,model):
        self.rawData=rawData
        df=self.rawData
        self.model=model
        rf_clf=self.model
        data=df.copy()
        print(df.columns)
        ppobjtd=preprocess.dataPreprocess()
        edsobjtd=EncodingandSplit.encodingandsplit()
        preProcessedData=ppobjtd.preprocesstestData(df)
        #features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU','Final Group', 'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the query ', 'Lastest RAG Status']
        #features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU','FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query', 'Lastest RAG Status','Reason for the RAG Status ']
        features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU', 'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the RAG Status ']
        df_rf_le = preProcessedData[features]
        df_rf_le=edsobjtd.labelEncoding(df_rf_le)
        predictions=rf_clf.predict(df_rf_le)
        probability=rf_clf.predict_proba(df_rf_le) 
        #data['Joining Status']=predictions
        data['Joining Probability']=probability[:,0]
        data['Rejection Probability']=probability[:,1]
        data['Joining Probability']= data['Joining Probability'].apply(lambda x: x*100)
        data['Rejection Probability']= data['Rejection Probability'].apply(lambda x: x*100)
        def condition(x):
            if x>=80:
                return "Green"
            elif x>=40 and x<80:
                return "Amber"
            else:
                return 'Red'
        # Applying the conditions
        data['Predictions'] = data['Joining Probability'].apply(condition)

        return data
         
    def results(self,rf_clf,X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf):
        self.rf_clf=rf_clf
        self.X_train_rf_clf=X_train_rf_clf
        X_train_rf_clf=self.X_train_rf_clf
        self.X_test_rf_clf=X_test_rf_clf
        X_test_rf_clf=self.X_test_rf_clf
        self.y_train_rf_clf=y_train_rf_clf
        y_train_rf_clf=self.y_train_rf_clf
        self.y_test_rf_clf=y_test_rf_clf
        y_test_rf_clf=self.y_test_rf_clf
        print("For Training")
        y_pred_train_rf_clf = rf_clf.predict(X_train_rf_clf)
        cm_train_rf_clf = confusion_matrix(y_train_rf_clf, y_pred_train_rf_clf)
        train_cm = ConfusionMatrixDisplay(cm_train_rf_clf,display_labels=["Join","Reject"]).plot() 
        plt.show()
        print("For Testing")
        y_pred_test_rf_clf = rf_clf.predict(X_test_rf_clf)
        cm_test_rf_clf = confusion_matrix(y_test_rf_clf,y_pred_test_rf_clf)
        test_cm = ConfusionMatrixDisplay(cm_test_rf_clf,display_labels=["Join","Reject"]).plot()
        plt.show()
        average = 'weighted'
        print("-----On Train Data--------")
        print(f"Precision:{precision_score(y_train_rf_clf,y_pred_train_rf_clf,average=average)}")
        print(f"Recall:{recall_score(y_train_rf_clf,y_pred_train_rf_clf,average=average)}")
        print(f"F1 Score:{f1_score(y_train_rf_clf,y_pred_train_rf_clf,average=average)}")
        y_pred_prob_train_rf_clf = rf_clf.predict_proba(X_train_rf_clf)
        #print(f"ROC_AUC_SCORE:{roc_auc_score(y_train_rf_clf, y_pred_prob_train_rf_clf, average=average, multi_class='ovr')}")
        #print(f"Log Loss:{log_loss(y_train_rf_clf,y_pred_prob_train_rf_clf)}")
        #log_loss(y_train_rf_clf,y_pred_prob_train_rf_clf)
        print(classification_report(y_train_rf_clf,y_pred_train_rf_clf,target_names=["Join","Reject"]))
        train_accuracy=accuracy_score(y_train_rf_clf,y_pred_train_rf_clf)
        print("\n------on Validation Data--------")
        print(f"Precision:{precision_score(y_test_rf_clf,y_pred_test_rf_clf,average=average)}")
        print(f"Recall:{recall_score(y_test_rf_clf,y_pred_test_rf_clf,average=average)}")
        print(f"F1 Score:{f1_score(y_test_rf_clf,y_pred_test_rf_clf,average=average)}")
        y_pred_prob_rf_clf = rf_clf.predict_proba(X_test_rf_clf)
        #print(f"ROC_AUC_SCORE:{roc_auc_score(y_test_rf_clf, y_pred_prob_rf_clf, average=average, multi_class='ovr')}")
        #print(f"Log Loss:{log_loss(y_test_rf_clf,y_pred_prob_rf_clf)}")
        #log_loss(y_test_rf_clf,y_pred_prob_rf_clf)
        print(classification_report(y_test_rf_clf,y_pred_test_rf_clf,target_names=["Join","Reject"]))
        test_accuracy=accuracy_score(y_test_rf_clf,y_pred_test_rf_clf)
        return train_accuracy,test_accuracy
    
    def Qualitycheck(self,rawData):
        self.rawData=rawData
        df=self.rawData
        Data=df.copy()
        print(Data.columns)
        ppobj=preprocess.dataPreprocess()
        edsobj=EncodingandSplit.encodingandsplit()
        preProcessedData=ppobj.preProcessing(self.rawData)
        #print(preProcessedData)
        #features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU','Final Group', #'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the query ', #'Lastest RAG Status']
        #features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU', #'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the RAG Status ', #'Lastest RAG Status']
        features=['Requisition City', 'Job Level', 'Job Profile', 'Business Group', 'Business Unit', 'Sub-BU', 'FinalDOJ_OfferAcceptanceDate', 'Final DOJ','Source', 'Type of Hire', 'Joining Period Slab', 'Type of Query','Reason for the RAG Status ']
        df_rf_le = preProcessedData[features]
        df_rf_le_y = preProcessedData['Joining Status']
        #print(df_rf_le_y.value_counts())
        df_rf_le=edsobj.labelEncoding(df_rf_le)
        print("10 Train Test Splits")
        Accuracy_scores=[]
        for seed in range(1,11):
            X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf=edsobj.traintestsplit(df_rf_le,df_rf_le_y,seed)
            rf_clf = RandomForestClassifier()
            rf_clf.fit(X_train_rf_clf,y_train_rf_clf)
            y_pred_test_rf_clf = rf_clf.predict(X_test_rf_clf)
            accuracy = accuracy_score(y_test_rf_clf,y_pred_test_rf_clf)
            Accuracy_scores.append(accuracy)
            print(accuracy)
        print("Cross validation ")
        kf=KFold(n_splits=10)
        score=cross_val_score(rf_clf,df_rf_le,df_rf_le_y,cv=kf)
        crossvalidation_scores=format(score)
        print("Cross Validation Scores :{}".format(score))
        print("Average Cross Validation score :{}".format(score.mean()))
        
        return Accuracy_scores,crossvalidation_scores
        
        