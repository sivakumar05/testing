from flask import Flask, request, url_for, redirect, render_template,send_from_directory,make_response
from Functions import classificationModel
import pandas as pd
from fileinput import filename

#<<<<<<< HEAD
#=======


#>>>>>>> 4ae8b0f1ae066ba8b7c071875fba712ab63808a8
#trainData = pd.read_excel('C:/HiringEngine/Data/RenegeData.xlsx')
#traindata=trainData.copy()
#testData = pd.read_excel('C:/HiringEngine/HiringEngine/RenegeAnalytics/RenegeAnalytics/Data/TestData.xlsx')
Finalmodel=pd.read_pickle('trained_model.pkl')

cmobj=classificationModel.classificationModel()
#rf_clf,X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf=cmobj.RF(trainData)
#df=cmobj.preProcesstestData(test_data[:2000],model)

app = Flask(__name__)

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
    resultdata=cmobj.preProcesstestData(testdata,Finalmodel)
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
