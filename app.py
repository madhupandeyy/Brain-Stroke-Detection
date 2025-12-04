# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
#import pandas as pd
import os
import requests 
import config
import pickle
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # prevents tkinter from being used as backend
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd 
# ==============================================================================================
# ==============================================================================================
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Load the dataset
#df = pd.read_csv('balanced_seizure_dataset_with_ids.csv')

# Drop unnamed index column if present
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Load the trained model
#with open("rf_model.pkl", "rb") as file:
#    loaded_model = pickle.load(file)



#disease_dic= ["Eye Spot","Healthy Leaf","Red Leaf Spot","Redrot","Ring Spot"]



from model_predict2  import pred_skin_disease

from model_predict2un import pred_skin_disease3

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page



#@ app.route('/')
#def home():
#    title = 'Multiple cancer Identification using Deeplearning'
#    return render_template('index.html', title=title)  
@app.route('/')
def home():
    return render_template('home1.html')        


 

@app.route('/patient',methods=['POST',"GET"])
def patient():
    return render_template('login44.html')    


@app.route('/admin',methods=['POST',"GET"])
def admin():
    return render_template('login442.html') 

@app.route('/register22',methods=['POST',"GET"])
def register22():
    return render_template('register442.html') 

@app.route('/register2',methods=['POST',"GET"])
def register2():
    return render_template('register44.html')  
import pickle
@app.route('/logedin',methods=['POST'])
def logedin():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
    

    name =int_features3[0]

    # Save to a file
    with open("name.pkl", "wb") as f:
        pickle.dump(name, f)

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('index.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               


@app.route('/logedin2',methods=['POST'])
def logedin2():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('patient_info.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               

import pandas as pd
from flask import request, render_template

@app.route('/get-patient-info', methods=['GET', 'POST'])
def get_patient_info():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')

        try:
            # Load the CSV file
            df = pd.read_csv('patient_data.csv')

            # Look for the row with matching patient ID
            row = df[df['patient_id'] == patient_id]

            if not row.empty:
                # Convert the row to dictionary (all columns)
                data = row.iloc[0].to_dict()
                
                # Optional: Join hospital lists if they are stored as comma-separated strings
                if 'india_hospitals' in data:
                    data['india_hospitals'] = str(data['india_hospitals'])
                if 'usa_hospitals' in data:
                    data['usa_hospitals'] = str(data['usa_hospitals'])

                return render_template('patient_info.html', data=data)
            else:
                return render_template('patient_info.html', error="Patient ID not found.")

        except Exception as e:
            return render_template('patient_info.html', error=f"Error: {str(e)}")

    # For GET request, just show the form
    return render_template('patient_info.html')


    

              
              # int_features3[0]==12345 and int_features3[1]==12345:
               #                      return render_template('index.html')
        
@app.route('/register',methods=['POST'])
def register():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login44.html')

@app.route('/register24',methods=['POST'])
def register24():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login442.html')                      
# render crop recommendation form page

from flask import request, render_template
from PIL import Image
import os
import pandas as pd
from datetime import datetime

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Brain Stroke Prediction using Deep Learning'

    if request.method == 'POST':
        file = request.files.get('file')

        # Load patient name/id from pickle
        with open("name.pkl", "rb") as f:
            patient_id = pickle.load(f)

        print("Patient ID:", patient_id)
        if not file or not patient_id:
            return render_template('rust.html', title=title)

        # Save uploaded image
        img = Image.open(file)
        img.save('output.png')

        # 3️⃣ Predict (your model functions)
        prediction2, confidencescore12 = pred_skin_disease3("output.png")
        print("Prediction 1:", prediction2)

        if prediction2 == "unknown":
            return render_template('error_page.html')

        # --- Main Stroke Prediction ---
        prediction, confidencescore = pred_skin_disease("output.png")  # replace with your actual model
        print("Prediction result:", prediction)

        # --- Detailed Stroke Information Database ---
        disease_info = {
            "stroke": {
                "cause": "Occurs when blood flow to part of the brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients.",
                "treatment": "Emergency care with clot-busting drugs, surgery, or rehabilitation depending on stroke type.",
                "homeopathy": "Arnica montana, Belladonna (supportive and preventive only).",
                "allopathy": "Thrombolytic therapy, antiplatelet or anticoagulant medication, carotid endarterectomy.",
                "ayurveda": "Panchakarma, Ashwagandha, Brahmi, and medicated oils for nerve recovery.",
                "india_hospitals": ["AIIMS Delhi", "Apollo Hospitals Chennai", "NIMHANS Bengaluru"],
                "usa_hospitals": ["Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital"],
                "cost_india": "₹1–3 lakhs (depending on severity and care type)",
                "cost_usa": "$15,000–30,000",
                "success_rate": "70–90% (depends on early diagnosis and care)"
            },
            "normal": {
                "cause": "No signs of blood flow obstruction or brain cell damage detected.",
                "treatment": "Maintain healthy lifestyle to prevent future risk — balanced diet, regular exercise, avoid smoking/alcohol.",
                "homeopathy": "N/A (preventive focus only).",
                "allopathy": "Regular check-ups, blood pressure and cholesterol control.",
                "ayurveda": "Meditation, yoga, and Rasayana herbs for brain health.",
                "india_hospitals": ["Fortis Hospital Delhi", "Apollo Chennai", "AIIMS Delhi"],
                "usa_hospitals": ["Mayo Clinic", "UCLA Health", "Mount Sinai Hospital"],
                "cost_india": "₹5,000–10,000 (preventive diagnostics)",
                "cost_usa": "$200–400",
                "success_rate": "100% (healthy condition)"
            }
        }

        # --- Get Predicted Disease Details ---
        prediction_type = prediction.lower().strip()
        details = disease_info.get(prediction_type, {})

        cause = details.get("cause", "Information not available.")
        treatment = details.get("treatment", "Information not available.")
        homeopathy = details.get("homeopathy", "Information not available.")
        allopathy = details.get("allopathy", "Information not available.")
        ayurveda = details.get("ayurveda", "Information not available.")
        india_hospitals = ", ".join(details.get("india_hospitals", []))
        usa_hospitals = ", ".join(details.get("usa_hospitals", []))
        cost_india = details.get("cost_india", "N/A")
        cost_usa = details.get("cost_usa", "N/A")
        success_rate = details.get("success_rate", "N/A")

        # --- CSV Database Update ---
        csv_file = 'patient_data.csv'
        columns = [
            'patient_id', 'brain_stroke_status', 'date', 'cause', 'treatment',
            'homeopathy', 'allopathy', 'ayurveda', 'india_hospitals',
            'usa_hospitals', 'cost_india', 'cost_usa', 'success_rate'
        ]

        # Create or load CSV
        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv(csv_file)

        current_date = datetime.now().strftime("%Y-%m-%d")

        new_data = {
            'patient_id': patient_id,
            'brain_stroke_status': prediction_type,
            'date': current_date,
            'cause': cause,
            'treatment': treatment,
            'homeopathy': homeopathy,
            'allopathy': allopathy,
            'ayurveda': ayurveda,
            'india_hospitals': india_hospitals,
            'usa_hospitals': usa_hospitals,
            'cost_india': cost_india,
            'cost_usa': cost_usa,
            'success_rate': success_rate
        }

        # Update or append
        if patient_id in df['patient_id'].values:
            idx = df[df['patient_id'] == patient_id].index[0]
            for key, value in new_data.items():
                df.at[idx, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        df.to_csv(csv_file, index=False)

        # --- Render Template with All Info ---
        return render_template(
            'rust-result.html',
            prediction=prediction.capitalize(),
            cause=cause,
            treatment=treatment,
            homeopathy=homeopathy,
            allopathy=allopathy,
            ayurveda=ayurveda,
            india_hospitals=india_hospitals,
            usa_hospitals=usa_hospitals,
            cost_india=cost_india,
            cost_usa=cost_usa,
            success_rate=success_rate,
            title="Brain Stroke Diagnosis Result"
        )

    return render_template('rust.html', title=title)












import matplotlib.pyplot as plt
import os

# Step 1: Define performance metrics
model_metrics = {
    "EfficientNet": {
        "accuracy": 0.9650,
        "precision": 0.9695,
        "recall": 0.9650,
        "f1_score": 0.9646
    },
    "ResNet50": {
        "accuracy": 0.9800,
        "precision": 0.9804,
        "recall": 0.9800,
        "f1_score": 0.9800
    },
    "MobileNet": {
        "accuracy": 0.9800,
        "precision": 0.9816,
        "recall": 0.9800,
        "f1_score": 0.9799
    }
}

# Step 2: Set output directory
output_dir = "static/img2"
os.makedirs(output_dir, exist_ok=True)

# Step 3: Prepare data
models = list(model_metrics.keys())
accuracies = [model_metrics[m]["accuracy"] for m in models]
precisions = [model_metrics[m]["precision"] for m in models]
recalls = [model_metrics[m]["recall"] for m in models]
f1_scores = [model_metrics[m]["f1_score"] for m in models]

# Step 4: Define function to plot each metric
def plot_metric(metric_values, metric_name):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, metric_values, color=['#4caf50', '#2196f3', '#ff9800'], edgecolor='black')
    plt.title(f"{metric_name} Comparison", fontsize=16)
    plt.ylabel(metric_name)
    plt.xlabel("Model")
    plt.ylim(0.9, 1.01)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f"{yval:.4f}", ha='center', fontsize=11)

    # Save figure
    filename = f"{metric_name.lower().replace(' ', '_')}_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Step 5: Generate and save all metric graphs
plot_metric(accuracies, "Accuracy")
plot_metric(precisions, "Precision")
plot_metric(recalls, "Recall")
plot_metric(f1_scores, "F1 Score")

print("✅ All metric comparison graphs saved to static/img2/")




# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
