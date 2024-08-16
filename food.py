from flask import Flask, request, redirect, url_for, render_template, flash, session, send_file, jsonify, send_from_directory
import pandas as pd
import pickle
import re
import secrets
import os
import firebase_admin
from firebase_admin import credentials, db
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from twilio.rest import Client
import subprocess
from dash_app import app as dash_app
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'your_secret_key'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Firebase
cred = credentials.Certificate('serviceAccountKeys.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://riskprediction-96c7f-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

ref = db.reference('patients_risk')
counter_ref = db.reference('counters')

# Load the trained model
model = pickle.load(open(r'C:\Users\DEEPIKA\OneDrive\Desktop\New folder\java\trained_ensemble_model.pkl', 'rb'))

# Load the dataset with Patient_IDs
df = pd.read_csv(r"C:\Users\DEEPIKA\OneDrive\Desktop\New folder\java\updated_dataset_with_ids.csv")
hf = pd.read_csv(r"C:\Users\DEEPIKA\OneDrive\Desktop\New folder\java\updated_dataset_with_ids.csv")


# Define the threshold for predicting readmission
THRESHOLD = 0.5
thresholds = {
    'Number_Past_Hospital_Admissions': 1.0,
    'Number_Past_ED_Visits_Total': 1.0,
    'Charlson_Comorbidity_Index': 3.0,
    'Number_Medications': 38.0,
    'Current_LOS_Groups': 7.5,
    'Dx_Cancer': 0.5,
    'Dx_Renal_Failure': 0.5,
    'Dx_Electrolyte_Disorder': 0.5,
    'Dx_Deficiency_Anemia': 0.5
}

def get_risk_causes(patient_data):
    risk_causes = []
    
    # Use calculated thresholds to identify risk causes
    if patient_data['Number_Past_Hospital_Admissions'].values[0] > thresholds['Number_Past_Hospital_Admissions']:
        risk_causes.append("High Number of Past Hospital Admissions")
    if patient_data['Number_Past_ED_Visits_Total'].values[0] > thresholds['Number_Past_ED_Visits_Total']:
        risk_causes.append("High Number of Emergency Department Visits")
    if patient_data['Charlson_Comorbidity_Index'].values[0] > thresholds['Charlson_Comorbidity_Index']:
        risk_causes.append("High Charlson Comorbidity Index")
    if patient_data['Number_Medications'].values[0] > thresholds['Number_Medications']:
        risk_causes.append("High Number of Medications")
    if patient_data['Current_LOS_Groups'].values[0] > thresholds['Current_LOS_Groups']:
        risk_causes.append("Extended Length of Stay")
    
    if patient_data['Dx_Cancer'].values[0] > thresholds['Dx_Cancer']:
        risk_causes.append("Cancer Diagnosis")
    if patient_data['Dx_Renal_Failure'].values[0] > thresholds['Dx_Renal_Failure']:
        risk_causes.append("Renal Failure Diagnosis")
    if patient_data['Dx_Electrolyte_Disorder'].values[0] > thresholds['Dx_Electrolyte_Disorder']:
        risk_causes.append("Electrolyte Disorder Diagnosis")
    if patient_data['Dx_Deficiency_Anemia'].values[0] > thresholds['Dx_Deficiency_Anemia']:
        risk_causes.append("Deficiency Anemia Diagnosis")
    
    # Return risk causes
    return risk_causes

def predict_for_patient(patient_id):
    patient_data = hf[hf['PatientID'] == patient_id]
    
    if patient_data.empty:
        return None
    
    patient_features = patient_data.drop(['Outcome_Readmission_within_30days', 'PatientID'], axis=1)
    probability = model.predict_proba(patient_features)[0][1] 

    predicted_readmission = 'Yes' if probability >= THRESHOLD else 'No'

    risk_causes = get_risk_causes(patient_data)

    result = {
        'Patient_ID': patient_id,
        'Predicted_Readmission': predicted_readmission,
        'Readmission_Probability': probability,
        'Risk_Causes': risk_causes
    }

    return result

def sanitize_column_names(df):
    sanitized_columns = []
    for col in df.columns:
        sanitized_col = col.replace('$', '_').replace('#', '_').replace('[', '_').replace(']', '_').replace('/', '_').replace('.', '_')
        sanitized_columns.append(sanitized_col)
    df.columns = sanitized_columns
    return df

def sanitize_data(df):
    df = df.applymap(lambda x: None if pd.isna(x) else x)
    return df

def generate_custom_key():
    current_counter = counter_ref.get() or 0
    new_counter = current_counter + 1
    counter_ref.set(new_counter)
    return f"PID{new_counter:03d}"

def validate_data(df):
    required_columns = [
        'Number_Past_ED_Visits_Total',
        'Treatment_type',
        'On_Corticosteroids_Total',
        'Charlson_Comorbidity_Index',
        'Dx_Renal_Failure',
        'Number_Past_Hospital_Admissions',
        'Number_Medications',
        'Age',
        'Current_LOS_Groups',
        'Residence_after_discharge'
    ]
    errors = []

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f'Missing columns: {", ".join(missing_columns)}')

    # Check for null values in required columns
    for index, row in df.iterrows():
        for col in required_columns:
            if pd.isna(row.get(col)):
                errors.append(f"Row {index} has null value for '{col}'")

    return errors
insurance_mapping = {1: 'General', 2: 'Semi-private', 3: 'Private'}

def get_dos_and_donts(disease):
    dos_and_donts = {
        'Cancer': {
            'dos': ['Follow the prescribed treatment plan', 'Maintain a healthy diet', 'Attend all follow-up appointments'],
            'donts': ['Ignore any new symptoms', 'Skip medications', 'Neglect mental health']
        },
        'Electrolyte': {
            'dos': ['Stay hydrated', 'Follow dietary recommendations', 'Regularly monitor electrolyte levels'],
            'donts': ['Consume excessive salt', 'Ignore symptoms like weakness or confusion', 'Miss follow-up appointments']
        },
        'Deficiency_Anemia': {
            'dos': ['Take iron supplements as prescribed', 'Eat iron-rich foods', 'Get regular blood tests'],
            'donts': ['Ignore fatigue or weakness', 'Consume tea or coffee with meals', 'Skip follow-up appointments']
        },
        'Renal': {
            'dos': ['Follow a kidney-friendly diet', 'Take medications as prescribed', 'Regularly monitor kidney function'],
            'donts': ['Consume high potassium foods', 'Ignore swelling or changes in urination', 'Miss dialysis sessions if prescribed']
        },
        'Drug_Abuse': {
            'dos': ['Seek professional help', 'Attend support groups', 'Adopt a healthy lifestyle'],
            'donts': ['Use drugs or alcohol', 'Isolate yourself', 'Ignore cravings or withdrawal symptoms']
        },
        # Add more diseases as needed
    }
    return dos_and_donts.get(disease, {
        'dos': ['Maintain a balanced diet', 'Stay active with regular exercise', 'Keep up with routine check-ups', 'Take medications as prescribed'],
        'donts': ['Avoid smoking and excessive alcohol', 'Do not skip medications', 'Avoid stress as much as possible']
    })
    
    # model1 = pickle.load(open(r'C:\Users\DEEPIKA\Downloads\readm fe\data\trained_ensemble_model.pkl', 'rb'))

# Load the dataset with Patient_IDs
df4= pd.read_csv(r'C:\Users\DEEPIKA\Downloads\readm fe\data\updated_dataset_with_ids.csv')


def predict_for_patient_for_2(patient_id):
    # Retrieve the data for the given PatientID
    patient_data = df4[df4['PatientID'] == patient_id]
    
    if patient_data.empty:
        return None

    # Determine the most likely disease for the patient (for demonstration, assume Dx_Cancer, etc. are binary indicators)
    diseases = ['Dx_Cancer', 'Dx_Electrolyte_Disorder', 'Dx_Deficiency_Anemia', 'Dx_Renal_Failure', 'Dx_Drug_Abuse']
    most_likely_disease = None
    severity = 'low'
    for disease in diseases:
        if patient_data[disease].values[0] == 1:
            most_likely_disease = disease.split('_')[1]
            # most_likely_disease = disease[1]
            if disease in ['Dx_Cancer', 'Dx_Renal_Failure']:
                severity = 'high'
            elif disease in ['Dx_Electrolyte_Disorder', 'Dx_Deficiency_Anemia']:
                severity = 'medium'
            else:
                severity = 'low'
            break
    print(f"Paitendid {patient_data}")
    print(f"disease {most_likely_disease}")

    dos_and_donts = get_dos_and_donts(most_likely_disease)

    # Check for future appointments
    future_appointments_value = patient_data['Future_Scheduled_Appointments'].values[0]
    try:
        future_appointments = int(future_appointments_value)
        has_future_appointments = future_appointments > 0
    except (ValueError, TypeError):
        has_future_appointments = False

    # Determine current insurance
    try:
        current_insurance = insurance_mapping[int(patient_data['Insurance'].values[0])]
    except (ValueError, TypeError, KeyError):
        current_insurance = "Unknown"
    
    # Insurance suggestions removed and replaced with given links
    insurance_suggestions = [
        'Ayushman Bharat National Health Protection Mission: https://www.india.gov.in/spotlight/ayushman-bharat-national-health-protection-mission',
        'Senior Citizen Health Insurance Scheme: https://www.policybazaar.com/health-insurance/senior-citizen-health-insurance/articles/government-plans-health-insurance-scheme-for-senior-citizens/',
        'DWBDNC Health Insurance: https://dwbdnc.dosje.gov.in/content/health-insurance',
        'Rashtriya Swasthya Bima Yojana: https://www.india.gov.in/spotlight/rashtriya-swasthya-bima-yojana'
    ]

    result = {
        'Patient_ID': patient_id,
        'Do\'s': dos_and_donts['dos'],
        'Don\'ts': dos_and_donts['donts'],
        'Has_Future_Appointments': has_future_appointments,
        'Current_Insurance': current_insurance,
        'Insurance_Suggestions': insurance_suggestions
    }
    
    return result


def validate_password(password):
    # Check password length
    if len(password) < 8:
        return False
    # Check for at least one number
    if not re.search(r"\d", password):
        return False
    # Check for at least one special character
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True


@app.route('/')
def home():
    return render_template('desktop_1.html')



@app.route('/desktop_2')
def desktop2():
    return render_template('desktop_2.html')

@app.route('/desktop_3')
def desktop3():
    return render_template('desktop_3.html')

@app.route('/desktop_4', methods=['GET', 'POST'])
def desktop4():
    error = None
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        password = request.form['password']
        
        # Check if patient ID exists
        patient_data = df[df['PatientID'] == patient_id]
        if patient_data.empty:
            error = "Patient ID not found."
        else:
            # Validate password
            if not validate_password(password):
                error = "Password must be at least 8 characters long, contain at least one number, and one special character."
            else:
                result = predict_for_patient_for_2(patient_id)
                if result:
                    # Convert the boolean values to Python's built-in bool type
                    result['Has_Future_Appointments'] = bool(result['Has_Future_Appointments'])
                    session['result'] = result
                    return redirect(url_for('result'))  # Ensure this matches the endpoint name
    return render_template('desktop_4.html', error=error)

logging.basicConfig(level=logging.DEBUG)

# Define your mappings here
mappings = {
    'Residence_before_admission': {
        1: 'At home', 3: 'Nursing home', 6: 'Other hospital (acute hospital) or Birth center',
        4: 'Old people\'s home', 8: 'Other', 2: 'At home with SPITEX Care', 83: 'Rehabilitation clinic'
    },
    'Admission_type': {
        1: 'Emergency (treatment within 12 hours indispensable)',
        2: 'Registered planned',
        5: 'Transfer within 24 hrs.'
    },
    'Referring_party': {
        2: 'Rescue service (ambulance police)',
        3: 'Doctor',
        1: 'Self relatives',
        8: 'Other',
        9: 'Unknown',
        6: 'Judicial authorities',
        4: 'Non-medical therapist',
        5: 'Social medical service'
    },
    'Treatment_type': {
        3: 'Inpatient'
    },
    'Insurance': {
        1: 'General', 2: 'Semi-private', 3: 'Private'
    },
    'Discharge_decision': {
        1: 'On the initiative of the Treating',
        3: 'On the initiative of a third person',
        2: 'On the initiative of the patient (against the opinion of the treating physician)',
        8: 'Other',
        9: 'Unknown'
    },
    'Residence_after_discharge': {
        1: 'Home', 2: 'Nursing home', 3: 'Old people\'s home', 6: 'Other hospital',
        8: 'Other', 4: 'Psychiatric hospital', 5: 'Rehabilitation clinic', 7: 'Penal institution'
    },
    'Care_after_discharge': {
        2: 'Outpatient treatment',
        4: 'Inpatient treatment or care',
        1: 'Cured/no Need for treatment',
        3: 'Outpatient care (e.g. SPITEX)',
        8: 'Other',
        5: 'Rehabilitation (amb. or stat.)',
        9: 'Unknown'
    },
    'Imaging_Orders': {1: 'Imaging ordered', 0: 'No imaging ordered'},
    'ECG_Order': {1: 'ECG ordered', 0: 'No ECG ordered'},
    'Restraining_Order': {0: 'No restraining order'},
    'Charlson_Comorbidity_Index': {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
        8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16'
    },
    'Future_Scheduled_Appointments': {1: 'Future appointments scheduled', 0: 'None scheduled'},
    'Ten_Days_LOS_Prior_Stay': {1: 'Length of stay over 10 days in prior stay', 0: 'No'},
    'Dx_Cancer': {1: 'Diagnosis of cancer', 0: 'No diagnosis'},
    'Dx_Electrolyte_Disorder': {1: 'Diagnosis of electrolyte disorder', 0: 'No diagnosis'},
    'Dx_Deficiency_Anemia': {1: 'Diagnosis of deficiency anemia', 0: 'No diagnosis'},
    'Dx_Renal_Failure': {1: 'Diagnosis of renal failure', 0: 'No diagnosis'},
    'Dx_Drug_Abuse': {1: 'Diagnosis of drug abuse', 0: 'No diagnosis'},
    'Hemoglobin_Low_Total': {1: 'Low hemoglobin', 0: 'Normal hemoglobin'},
    'Calcium_Low_Total': {1: 'Low calcium', 0: 'Normal calcium'},
    'BUN_High_Total': {1: 'High BUN', 0: 'Normal BUN'},
    'Creatinine_High_Total': {1: 'High creatinine', 0: 'Normal creatinine'},
    'PO4_Tested_Total': {1: 'PO4 tested', 0: 'Not tested'},
    'INR_High_TOTAL': {1: 'High INR', 0: 'Normal INR'},
    'On_Anticoagulants_Total': {1: 'On anticoagulants', 0: 'Not on anticoagulants'},
    'On_NSAIDS_Total': {1: 'On NSAIDs', 0: 'Not on NSAIDs'},
    'On_Corticosteroids_Total': {1: 'On corticosteroids', 0: 'Not on corticosteroids'},
    'On_Corticosteroids_Total_Mild': {1: 'On mild corticosteroids', 0: 'Not on mild corticosteroids'},
    'On_Antipsychotics_Total': {1: 'On antipsychotics', 0: 'Not on antipsychotics'},
    'On_Ulcer_Medications_Total': {1: 'On ulcer medications', 0: 'Not on ulcer medications'}
}

# Load data from CSV file
df = pd.read_csv(r'C:\Users\DEEPIKA\Downloads\readm fe\data\updated_dataset_with_ids.csv')

# Apply the mappings
for column, mapping in mappings.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

def generate_pdf(patient_id, df):
    patient_data = df[df['PatientID'] == patient_id]

    if patient_data.empty:
        return None

    patient_data = patient_data.iloc[0]

    pdf_path = f"Patient_Report_{patient_id}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    margin = 1 * inch  # 1-inch margin
    y_position = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "Patient Report")
    c.setFont("Helvetica-Bold", 10)

    fields = [
        ("Patient ID", patient_data['PatientID']),
        ("Age", patient_data['Age']),
        ("Residence Before Admission", patient_data['Residence_before_admission']),
        ("Admission Type", patient_data['Admission_type']),
        ("Referring Party", patient_data['Referring_party']),
        ("Treatment Type", patient_data['Treatment_type']),
        ("Insurance", patient_data['Insurance']),
        ("Discharge Decision", patient_data['Discharge_decision']),
        ("Residence After Discharge", patient_data['Residence_after_discharge']),
        ("Care After Discharge", patient_data['Care_after_discharge']),
        ("Case Mix Effective", patient_data['Case_Mix_Effective']),
        ("Current LOS Groups", patient_data['Current_LOS_Groups']),
        ("Number of Past Hospital Admissions", patient_data['Number_Past_Hospital_Admissions']),
        ("Number of Past ED Visits Subsequent Hospital Stay", patient_data['Number_Past_ED_Visits_Subsequent_HospitalStay']),
        ("Number of Past ED Ambulant Visits", patient_data['Number_Past_ED_AmbulantVisits']),
        ("Number of Past ED Visits Total", patient_data['Number_Past_ED_Visits_Total']),
        ("Future Scheduled Appointments", patient_data['Future_Scheduled_Appointments']),
        ("Ten Days LOS Prior Stay", patient_data['Ten_Days_LOS_Prior_Stay']),
        ("Diagnosis Cancer", patient_data['Dx_Cancer']),
        ("Diagnosis Electrolyte Disorder", patient_data['Dx_Electrolyte_Disorder']),
        ("Diagnosis Deficiency Anemia", patient_data['Dx_Deficiency_Anemia']),
        ("Diagnosis Renal Failure", patient_data['Dx_Renal_Failure']),
        ("Diagnosis Drug Abuse", patient_data['Dx_Drug_Abuse']),
        ("Hemoglobin Low Total", patient_data['Hemoglobin_Low_Total']),
        ("Calcium Low Total", patient_data['Calcium_Low_Total']),
        ("BUN High Total", patient_data['BUN_High_Total']),
        ("Creatinine High Total", patient_data['Creatinine_High_Total']),
        ("PO4 Tested Total", patient_data['PO4_Tested_Total']),
        ("INR High Total", patient_data['INR_High_TOTAL']),
        ("On Anticoagulants Total", patient_data['On_Anticoagulants_Total']),
        ("On NSAIDS Total", patient_data['On_NSAIDS_Total']),
        ("On Corticosteroids Total", patient_data['On_Corticosteroids_Total']),
        ("On Corticosteroids Total Mild", patient_data['On_Corticosteroids_Total_Mild']),
        ("On Antipsychotics Total", patient_data['On_Antipsychotics_Total']),
        ("On Ulcer Medications Total", patient_data['On_Ulcer_Medications_Total']),
    ]

    for field, value in fields:
        y_position -= 20
        c.drawString(margin, y_position, f"{field}: {value}")

    c.save()
    return pdf_path

# Function to extract the file ID from a Google Drive link
def extract_file_id(drive_link):
    import re
    match = re.search(r'/d/([^/]+)', drive_link)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid Google Drive link")

# Function to create a direct download link
def create_direct_download_link(file_id):
    return f'https://drive.google.com/uc?export=download&id={file_id}'

def upload_file_to_google_drive(file_path):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    SERVICE_ACCOUNT_FILE = 'pdf-upload-431204-219655e9edc5.json'
    folder_id = '1Dqs1aMdTRdO5qCC8EuYnPyUc_H_9p-pn'  # Add your Google Drive folder ID here

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('drive', 'v3', credentials=credentials)

    # File metadata including the folder ID
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]  # Specify the folder where the file will be uploaded
    }
    media = MediaFileUpload(file_path, mimetype='application/pdf')

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    # Make the file publicly accessible
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(fileId=file_id, body=permission).execute()

    # Create the shareable link and convert to direct download link
    file_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    direct_download_link = create_direct_download_link(file_id)
    return direct_download_link

# Function to send the PDF report via SMS
def send_pdf_via_sms(direct_download_link, to_number):
    account_sid = 'AC719484270b93cbdf0ad9ad4c94677b92'
    auth_token = '581cfcb1080fd99cd4e3b1f821148957'
    client = Client(account_sid, auth_token)

    from_phone_number = '+18576880428'  # Replace with your Twilio phone number
    to_phone_number = to_number

    message = client.messages.create(
        body='Your patient report is ready. Please find the attached PDF.',
        from_=from_phone_number,
        to=to_phone_number,
        media_url=[direct_download_link]  # Use the direct download URL here
    )
    return message.sid


@app.route('/result')
def result():
    result = session.get('result', None)
    if result:
        return render_template('desktop_8_food.html', result=result)
    else:
        return redirect(url_for('desktop_4'))


@app.route('/predict', methods=['POST'])
def predict():
    patient_id = request.form['patient_id']
    result = predict_for_patient(patient_id)
    if result:
        return render_template('result.html', result=result)
    else:
        return "Patient ID not found.", 404


@app.route('/desktop_5')
def desktop5():
    return render_template('desktop_5.html')

@app.route('/desktop_6')
def desktop6():
    return render_template('desktop_6.html')

@app.route('/desktop_7')
def desktop7():
    return render_template('desktop_7.html')

@app.route('/run-day-script')
def run_day_script():
    # Call day.py script
    subprocess.Popen(['python', 'dash_app.py'])
    # Redirect back to the dashboard or any other page
    return redirect(url_for('dashboard'))

@app.route('/dash')
def dashboard():
    return "Overall Risk Dashboard"  

@app.route('/add_patient')
def add_patient():
    return render_template('add_patient.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('add_patient'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('add_patient'))

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            new_data = pd.read_csv(filepath)
            new_data = sanitize_column_names(new_data)
            new_data = sanitize_data(new_data)

            # Validate the data
            validation_errors = validate_data(new_data)
            if validation_errors:
                error_messages = "\n".join(validation_errors)
                flash(f'Data validation failed:\n{error_messages}', 'error')
                return redirect(url_for('add_patient'))

            # Convert DataFrame to a dictionary of records
            records = new_data.to_dict(orient='records')

            # Upload each record to Firebase with a custom key
            for record in records:
                custom_key = generate_custom_key()
                ref.child(custom_key).set(record)

            flash('File uploaded and data updated successfully!', 'success')
            return redirect(url_for('add_patient'))
        except Exception as e:
            flash(f'Error occurred: {str(e)}', 'error')
            return redirect(url_for('add_patient'))
    
    flash('Invalid file type', 'error')
    return redirect(url_for('add_patient'))

# Route to render whatsapp.html
@app.route('/whatsapp/')
def whatsapp():
    return render_template('whatsapp.html')

@app.route('/guide.html')
def guide():
    return render_template('guide.html')



@app.route('/generate_report', methods=['POST'])
def generate_report():
    patient_id = request.form['patient_id']
    phone_number = request.form['phone_number']
    
    if not patient_id.startswith("PID_") or not patient_id[4:].isdigit():
        return jsonify({'error': 'Invalid Patient ID. Please enter in the format PID_XXXXXX.'}), 400
    
    pdf_path = generate_pdf(patient_id, df)
    
    if not pdf_path:
        return jsonify({'error': 'No data found for the given Patient ID.'}), 404

    direct_download_link = upload_file_to_google_drive(pdf_path)
    send_pdf_via_sms(direct_download_link, phone_number)
    
    return jsonify({'pdf_url': pdf_path})

@app.route('/view_report/<path:pdf_path>')
def view_report(pdf_path):
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'File not found.'}), 404
    return send_file(pdf_path, as_attachment=False)

model1 = pickle.load(open(r'C:\Users\DEEPIKA\Downloads\readm fe\data\trained_ensemble_model.pkl', 'rb'))

# Load the dataset with Patient_IDs
df1 = pd.read_csv(r'C:\Users\DEEPIKA\Downloads\readm fe\data\updated_dataset_with_ids.csv')

diet_plans = {
    'Cancer': {
        'Day 1': {'Breakfast': 'Spinach Moong Dal Cheela with Protein Shake + Papaya', 'Lunch': 'Multigrain Roti along with Fish Curry and Greek Salad', 'Snacks': 'Pineapple', 'Dinner': 'Multigrain Roti & Mushroom Matar Paneer Curry, Tofu Vegetable Salad'},
        'Day 2': {'Breakfast': 'Overnight oats berries breakfast bowl + 1 Seasonal Fruit', 'Lunch': 'Brown rice, Lauki Chana Dal, Anda Curry, and Chana Salad', 'Snacks': 'Dry fruits & seeds', 'Dinner': 'Quinoa vegetable Khichdi, shahi paneer, vegetable salad'},
        'Day 3': {'Breakfast': 'Scrambled egg spread on toast, Protein Shake', 'Lunch': 'Grilled Chicken with White Bean & Tomato Salad', 'Snacks': 'Vegetable sticks', 'Dinner': 'Broiled Fish, Brown rice, Mixed Green Salad'},
        'Day 4': {'Breakfast': 'Almond/soy milk + avocado toast', 'Lunch': 'Lean chicken burger with lettuce, tomato & beans', 'Snacks': 'Bananas', 'Dinner': 'Palak paneer, multigrain roti, Lauki raita, mango salad'},
        'Day 5': {'Breakfast': 'Boiled egg with steamed vegetables', 'Lunch': 'Lemon gravy chicken, brown rice, Greek yoghurt', 'Snacks': 'Hard-boiled egg & carrot sticks', 'Dinner': 'Tofu curry, whole wheat roti, sprouts salad'},
        'Day 6': {'Breakfast': 'Raw paneer patty sandwich in whole grain bread, fruits', 'Lunch': 'Whole wheat chicken pasta with beans and green salad', 'Snacks': 'Vegetable sticks', 'Dinner': 'Stuffed baked mushroom, green salad, multi-grain toast'},
        'Day 7': {'Breakfast': 'Apple peanut butter smoothie with whey protein', 'Lunch': 'Multigrain chapati, egg bhurji, cucumber raita, sauteed vegetable', 'Snacks': 'Apple slices with nut butter', 'Dinner': 'Multigrain millet vegetable khichdi, curd, Greek salad'}
    },
    'Electrolyte': {
        'Day 1': {'Breakfast': 'Banana smoothie with chia seeds', 'Lunch': 'Grilled chicken with quinoa and vegetables', 'Snacks': 'Apple slices with peanut butter', 'Dinner': 'Baked salmon with steamed broccoli'},
        'Day 2': {'Breakfast': 'Greek yogurt with berries', 'Lunch': 'Turkey sandwich on whole grain bread', 'Snacks': 'Carrot sticks with hummus', 'Dinner': 'Stir-fried tofu with mixed vegetables'},
        'Day 3': {'Breakfast': 'Oatmeal with almonds and honey', 'Lunch': 'Lentil soup with a side salad', 'Snacks': 'Mixed nuts', 'Dinner': 'Chicken breast with sweet potatoes'},
        'Day 4': {'Breakfast': 'Smoothie bowl with granola', 'Lunch': 'Shrimp salad with avocado', 'Snacks': 'Orange slices', 'Dinner': 'Beef stir-fry with bell peppers'},
        'Day 5': {'Breakfast': 'Scrambled eggs with spinach', 'Lunch': 'Vegetable wrap with hummus', 'Snacks': 'Celery sticks with cream cheese', 'Dinner': 'Pasta with marinara sauce and veggies'},
        'Day 6': {'Breakfast': 'Whole grain toast with avocado', 'Lunch': 'Grilled fish tacos', 'Snacks': 'Protein bar', 'Dinner': 'Chicken stew with carrots and peas'},
        'Day 7': {'Breakfast': 'Yogurt parfait with fruits', 'Lunch': 'Quinoa salad with black beans', 'Snacks': 'Grapes', 'Dinner': 'Baked chicken with green beans'}
    },
    'Deficiency_Anemia': {
        'Day 1': {'Breakfast': 'Iron-fortified cereal with milk', 'Lunch': 'Spinach salad with chicken', 'Snacks': 'Dried apricots', 'Dinner': 'Beef stir-fry with bell peppers'},
        'Day 2': {'Breakfast': 'Whole grain toast with peanut butter', 'Lunch': 'Turkey and cheese sandwich', 'Snacks': 'Pumpkin seeds', 'Dinner': 'Lentil soup with a side of greens'},
        'Day 3': {'Breakfast': 'Oatmeal with raisins and nuts', 'Lunch': 'Quinoa salad with vegetables', 'Snacks': 'Apple slices', 'Dinner': 'Grilled chicken with steamed broccoli'},
        'Day 4': {'Breakfast': 'Scrambled eggs with spinach', 'Lunch': 'Beef and barley soup', 'Snacks': 'Sunflower seeds', 'Dinner': 'Fish curry with brown rice'},
        'Day 5': {'Breakfast': 'Smoothie with spinach and berries', 'Lunch': 'Chicken wrap with avocado', 'Snacks': 'Orange slices', 'Dinner': 'Baked tofu with mixed vegetables'},
        'Day 6': {'Breakfast': 'Yogurt with granola', 'Lunch': 'Salmon salad with chickpeas', 'Snacks': 'Pear slices', 'Dinner': 'Stir-fried beef with green beans'},
        'Day 7': {'Breakfast': 'Fruit salad with almonds', 'Lunch': 'Vegetable pasta with beans', 'Snacks': 'Mixed nuts', 'Dinner': 'Chicken and vegetable stew'}
    },
    'Renal': {
        'Day 1': {'Breakfast': 'Low-potassium fruit smoothie', 'Lunch': 'Grilled chicken with white rice', 'Snacks': 'Apple slices', 'Dinner': 'Baked fish with green beans'},
        'Day 2': {'Breakfast': 'Egg white omelette with vegetables', 'Lunch': 'Turkey and lettuce wrap', 'Snacks': 'Grapes', 'Dinner': 'Pasta with marinara sauce'},
        'Day 3': {'Breakfast': 'Cottage cheese with peaches', 'Lunch': 'Rice and bean salad', 'Snacks': 'Rice cakes with cream cheese', 'Dinner': 'Chicken stir-fry with vegetables'},
        'Day 4': {'Breakfast': 'Plain yogurt with berries', 'Lunch': 'Tuna salad on white bread', 'Snacks': 'Pear slices', 'Dinner': 'Baked chicken with carrots'},
        'Day 5': {'Breakfast': 'Smoothie with almond milk', 'Lunch': 'Shrimp with white rice', 'Snacks': 'Strawberries', 'Dinner': 'Beef and broccoli stir-fry'},
        'Day 6': {'Breakfast': 'Scrambled eggs with bell peppers', 'Lunch': 'Chicken noodle soup', 'Snacks': 'Cucumber slices', 'Dinner': 'Pork chops with green beans'},
        'Day 7': {'Breakfast': 'Oatmeal with blueberries', 'Lunch': 'Ham sandwich on white bread', 'Snacks': 'Melon slices', 'Dinner': 'Grilled fish with rice'}
    },
    'Drug_Abuse': {
        'Day 1': {'Breakfast': 'Green smoothie with spinach and apple', 'Lunch': 'Chicken and quinoa salad', 'Snacks': 'Mixed nuts', 'Dinner': 'Grilled salmon with asparagus'},
        'Day 2': {'Breakfast': 'Greek yogurt with honey and almonds', 'Lunch': 'Turkey and avocado wrap', 'Snacks': 'Carrot sticks with hummus', 'Dinner': 'Vegetable stir-fry with tofu'},
        'Day 3': {'Breakfast': 'Oatmeal with fresh berries', 'Lunch': 'Lentil and vegetable soup', 'Snacks': 'Apple slices with peanut butter', 'Dinner': 'Chicken breast with mixed vegetables'},
        'Day 4': {'Breakfast': 'Whole grain toast with avocado', 'Lunch': 'Salmon and kale salad', 'Snacks': 'Orange slices', 'Dinner': 'Beef stir-fry with bell peppers'},
        'Day 5': {'Breakfast': 'Smoothie with banana and almond milk', 'Lunch': 'Chicken wrap with veggies', 'Snacks': 'Celery sticks with cream cheese', 'Dinner': 'Pasta with marinara sauce and veggies'},
        'Day 6': {'Breakfast': 'Scrambled eggs with spinach', 'Lunch': 'Grilled shrimp with quinoa', 'Snacks': 'Protein bar', 'Dinner': 'Chicken stew with carrots and peas'},
        'Day 7': {'Breakfast': 'Yogurt parfait with granola', 'Lunch': 'Vegetable and chickpea salad', 'Snacks': 'Grapes', 'Dinner': 'Baked chicken with green beans'}
    }
}

def get_diet_plan(disease):
    # Return the diet plan if the disease is found in the dictionary, otherwise return None
    if disease is None or disease not in diet_plans:
        return None
    return diet_plans[disease]
def predict_for_patient_food(patient_id):
    # Retrieve the data for the given Patient_ID
    patient_data = df1[df1['PatientID'] == patient_id]
    
    if patient_data.empty:
        return None

    # Determine the most likely disease for the patient (for demonstration, assume Dx_Cancer, etc. are binary indicators)
    diseases = ['Dx_Cancer', 'Dx_Electrolyte_Disorder', 'Dx_Deficiency_Anemia', 'Dx_Renal_Failure', 'Dx_Drug_Abuse']
    most_likely_disease = None
    for disease in diseases:
        if patient_data[disease].values[0] == 1:
            most_likely_disease = disease.split('_')[1]
            break

    diet_plan = get_diet_plan(most_likely_disease)

    result = {
        'Patient_ID': patient_id,
        'Disease': most_likely_disease,
        'Diet_Plan': diet_plan if diet_plan else {}  # Ensure Diet_Plan is not None
    }
    
    return result

@app.route('/index', methods=['GET', 'POST'])
def index():
    error = None
    result = None
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        
        # Check if patient ID exists
        patient_data = df1[df1['PatientID'] == patient_id]
        if patient_data.empty:
            error = "Patient ID not found."
        else:
            result = predict_for_patient_food(patient_id)
    
    return render_template('index.html', error=error, result=result)

@app.route('/home_2')

def home_2():
    return render_template('desktop_1.html')

@app.route('/home_1')
def home_1():
    return render_template('desktop_1.html')

@app.route('/desktop_8')
def desktop_8():
    return render_template('desktop_8.html')
    

if __name__ == '__main__':
    app.run(debug=True)
