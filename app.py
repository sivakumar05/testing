from flask import Flask, request, url_for, redirect, render_template,send_from_directory,make_response
from Functions import classificationModel
import pandas as pd
from fileinput import filename
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

Finalmodel=pd.read_pickle('trained_model.pkl')

cmobj=classificationModel.classificationModel()

app = Flask(__name__)

def labelEncoding(self,data):
    label = LabelEncoder()
    
    LEdata=data
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
    
def condition(x):
    if x>=80:
        return "Green"
    elif x>=40 and x<80:
        return "Amber"
    else:
        return 'Red'


def preProcesstestData(rawData,model):
    
    df=rawData
    
    rf_clf=model
    data=df.copy()
    print(df.columns)
    ppobjtd=dataPreprocess()
    edsobjtd=encodingandsplit()
    preProcessedData=preprocesstestData(df)
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
    # Applying the conditions
    data['Predictions'] = data['Joining Probability'].apply(condition)

    return data
# Root endpoint
@app.get('/')
def upload():
	return render_template('index.html')
    

@app.post('/view')
def view():
 
    # Read the File using Flask request
    file = request.files['file']
    # save file in local directory
    file.save(file.filename)
 
    # Parse the data as a Pandas DataFrame type
    testdata = pd.read_excel(file)
    resultdata=preProcesstestData(testdata,Finalmodel)
    #resultdata.to_excel('resultdata.xlsx',index=False)
    
    resp = make_response(resultdata.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=Renege_Result.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.post('/template')
def template():
    resp1 = make_response(traindata.to_csv(index=False))
    resp1.headers["Content-Disposition"] = "attachment; filename=RawData_Template.csv"
    resp1.headers["Content-Type"] = "text/csv"
    return resp1


        
if __name__ =="__main__":
    app.run()
