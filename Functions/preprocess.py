# Importing the required libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

class dataPreprocess():
    
    def preProcessing(self,rawData):
        print("preProcessing rawdata...")
        self.rawData=rawData
        df=self.rawData

        #Dropping irrelevant columns
        df.drop(['Position ID', 'Job Requisition ID','Candidate ID','Revised Target Hire Date Proposed','Target Hire Date Proposed','Decline Confirmation Date','Last Employer','Skill','Total Experience','Combi'],axis=1,inplace=True)
        
        #Drop rows with  joining status "Joining In Pipeline" & "Offer Accepted, DOJ in the Past"
        df = df[(df['Joining Status'] !="Joining In Pipeline")]
        df = df[(df['Joining Status'] !="Offer Accepted, DOJ in the Past")]
        
        #replacing "Declined by candidate with Rejected values"
        df['Joining Status'] = np.where((df['Joining Status'] == 'Declined by Candidate'), 'Reject', df['Joining Status'])
        df['Joining Status'] = np.where((df['Joining Status'] == 'Rejected'), 'Reject', df['Joining Status'])
        df['Joining Status'] = np.where((df['Joining Status'] == 'Joined'), 'Join', df['Joining Status'])

        #Replacing "RPO Campus" with "Lateral"
        df['Type of Hire']=df['Type of Hire'].replace(['RPO Campus'], 'Lateral')

        #Filling Null Values in "Type of Query" column with data from "Final Group" & "Type of Hire"
        df['Type of Query'] = np.where((df['Type of Query'].isna()),'no Query available',df['Type of Query'])

        #Filling Null Values in "Source" column with "No Source available"
        df['Source'] = np.where(df['Source'].isna(),'No Source available', df['Source'])
        
         #Filling Null Values in "Decline Reasons List" column with data from "Joining Status"
        df['Decline Reasons List'] = np.where((df['Decline Reasons List'].isna()) & (df['Joining Status'] == 'Join'), 'Joined', df['Decline Reasons List'])
        # make the below 2 "no pofu available"
        df['Decline Reasons List'] = np.where((df['Decline Reasons List'].isna()) & (df['Joining Status'] == 'Reject'), 'No pofu available', df['Decline Reasons List'])


        #Filling Null Values in "Lastest RAG Status" column "No Rag status Available"
        df['Lastest RAG Status'] = np.where((df['Lastest RAG Status'].isna()),'no RAG status available',df['Lastest RAG Status'])

        #Filling Null Values in "Reason for the RAG status" column with data from "Reason for Red Category - RAG", "Reason for Yellow Category - RAG" & "Joining Status"
        df['Reason for the RAG Status '] = "RAG"
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") &(df['Lastest RAG Status']=='Red'), df['Reason for Red Category - RAG'], df['Reason for the RAG Status '])
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") &(df['Lastest RAG Status']=='Yellow'), df['Reason for Yellow Category - RAG'], df['Reason for the RAG Status '])
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") &(df['Lastest RAG Status']=='Green'),'RAG Status-Green', df['Reason for the RAG Status '])
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") ,"No RAG status available", df['Reason for the RAG Status '])


        #Filling Null Values in "Reason for the query" column with data from "Final Group" , "Type of Hire" & Joining Status
        df['Reason for the query '] = np.where(df['Reason for the query '].isna(),'No Reason for the Query available', df['Reason for the query '])

        #Dropping Reason for Red Category - RAG & Reason for Yellow Category - RAG columns
        df.drop(['Reason for Red Category - RAG','Reason for Yellow Category - RAG'],axis=1,inplace=True)

        #Fill missing values with "no gender specified"
        df['Gender'] = np.where(df['Gender'].isna(), 'no gender specified', df['Gender'])

        #dropping any left over rows with missing values
        df = df.dropna(how='any',axis=0)

        #Calculate days Difference between columns
        df['FinalDOJ_OfferAcceptanceDate'] = (df['Final DOJ'] - df['Offer Acceptance Date and time Stamp']).dt.days
        
        return df
    def preprocesstestData(self,rawData):
        print("preProcessing rawdata...")
        self.rawData=rawData
        df=self.rawData
        
        #Dropping irrelevant columns
        df.drop(['Position ID', 'Job Requisition ID','Candidate ID','Revised Target Hire Date Proposed','Target Hire Date Proposed','Decline Confirmation Date','Last Employer','Skill','Total Experience','Combi','Decline Reasons List'],axis=1,inplace=True)

        #Replacing "RPO Campus" with "Lateral"
        df['Type of Hire']=df['Type of Hire'].replace(['RPO Campus'], 'Lateral')

        #Filling Null Values in "Type of Query" column with data from "Final Group" & "Type of Hire"
        df['Type of Query'] = np.where((df['Type of Query'].isna()),'no Query available',df['Type of Query'])

        #Filling Null Values in "Source" column with "No Source available"
        df['Source'] = np.where(df['Source'].isna(),'No Source available', df['Source'])

        #Filling Null Values in "Lastest RAG Status" column with data from "Final Group" & "Type of Hire"
        df['Lastest RAG Status'] = np.where((df['Lastest RAG Status'].isna()),'no RAG status available',df['Lastest RAG Status'])

        #Filling Null Values in "Reason for the query" column with data from "Final Group" , "Type of Hire" & Joining Status
        df['Reason for the query '] = np.where(df['Reason for the query '].isna(),'No Reason for the Query avaiilable', df['Reason for the query '])

        #Filling Null Values in "Reason for the RAG status" column with data from "Reason for Red Category - RAG", "Reason for Yellow Category - RAG" & "Joining Status"
        df['Reason for the RAG Status '] = "RAG"
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") &(df['Lastest RAG Status']=='Red'), df['Reason for Red Category - RAG'], df['Reason for the RAG Status '])
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") &(df['Lastest RAG Status']=='Yellow'), df['Reason for Yellow Category - RAG'], df['Reason for the RAG Status '])
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") &(df['Lastest RAG Status']=='Green'),'RAG Status-Green', df['Reason for the RAG Status '])
        df['Reason for the RAG Status '] = np.where((df['Reason for the RAG Status '] =="RAG") ,"No RAG status available", df['Reason for the RAG Status '])

        #Fill missing values with "no gender specified"
        df['Gender'] = np.where(df['Gender'].isna(), 'no gender specified', df['Gender'])

        #Calculate days Difference between columns
        df['FinalDOJ_OfferAcceptanceDate'] = (df['Final DOJ'] - df['Offer Acceptance Date and time Stamp']).dt.days
        #print(df.isna().sum())
        
        return df
    def plot_feature_importance(self,importance,names,model_type):
        self.importance=importance
        self.names=names
        self.model_type=model_type
        #Create arrays from feature importance and feature names
        feature_importance = np.array(self.importance)
        feature_names = np.array(self.names)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        #Add chart labels
        plt.title(model_type + ' FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')   
        plt.ylabel('FEATURE NAMES')