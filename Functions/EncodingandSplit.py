# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

class encodingandsplit():
    def labelEncoding(self,data):
        label = LabelEncoder()
        self.data=data
        LEdata=self.data
        LEdata.loc[:, 'Requisition City'] = label.fit_transform(LEdata['Requisition City'])
        LEdata.loc[:, 'Job Level'] = label.fit_transform(LEdata['Job Level'])
        LEdata.loc[:, 'Job Profile'] = label.fit_transform(LEdata['Job Profile'])
        LEdata.loc[:, 'Business Group'] = label.fit_transform(LEdata['Business Group'])
        LEdata.loc[:, 'Business Unit'] = label.fit_transform(LEdata['Business Unit'])
        LEdata.loc[:, 'Sub-BU'] = label.fit_transform(LEdata['Sub-BU'])
        LEdata.loc[:, 'Source'] = label.fit_transform(LEdata['Source'])
        LEdata.loc[:, 'Type of Hire'] = label.fit_transform(LEdata['Type of Hire'])
        LEdata.loc[:, 'Type of Query'] = label.fit_transform(LEdata['Type of Query'])
        #LEdata.loc[:, 'Reason for the query '] = label.fit_transform(LEdata['Reason for the query '])
        #LEdata.loc[:, 'Lastest RAG Status'] = label.fit_transform(LEdata['Lastest RAG Status'])
        #LEdata.loc[:, 'Final Group'] = label.fit_transform(LEdata['Final Group'])
        LEdata.loc[:, 'Joining Period Slab'] = label.fit_transform(LEdata['Joining Period Slab'])
        LEdata.loc[:, 'Final DOJ'] = label.fit_transform(LEdata['Final DOJ'])
        LEdata.loc[:, 'Reason for the RAG Status '] = label.fit_transform(LEdata['Reason for the RAG Status '])
        return LEdata
    def traintestsplit(self,X,Y,seed):
        self.X=X
        self.Y=Y
        self.seed=seed
        X_train_clf,X_test_clf,y_train_clf,y_test_clf = train_test_split(self.X,self.Y,test_size=0.25,
                                                                             random_state=self.seed,shuffle=True,
                                                                             stratify=self.Y)
        return X_train_clf,X_test_clf,y_train_clf,y_test_clf