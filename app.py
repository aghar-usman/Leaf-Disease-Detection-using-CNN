import numpy as np
import os
import shutil
import cv2
import sqlite3
import tflearn
import tensorflow as tf
from random import shuffle
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import requests
import json
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = '1122'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ThingSpeak Configuration
THINGSPEAK_CHANNEL_ID = '2924568'
THINGSPEAK_READ_API_KEY = 'KZLM0IFPQD80E5N1'
THINGSPEAK_FIELDS = {
    'Moisture': 1,
    'Temperature': 2,
    'Humidity': 3,
    'N-value': 4,
    'P-value': 5,
    'K-value': 6
}

# Database initialization
def init_db():
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user(
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            mobile TEXT,
            email TEXT
        )
    """)
    connection.commit()
    connection.close()

init_db()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_thingspeak_data(field, days=7):
    """Fetch data from ThingSpeak for a specific field over a number of days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/fields/{field}.json"
    params = {
        'api_key': THINGSPEAK_READ_API_KEY,
        'start': start_date.strftime('%Y-%m-%d %H:%M:%S'),
        'end': end_date.strftime('%Y-%m-%d %H:%M:%S'),
        'days': days,
        'round': 2,
        'timescale': 60
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'feeds' in data:
            # Extract values and timestamps
            values = []
            days = []
            for entry in data['feeds']:
                field_key = f'field{field}'
                if field_key in entry and entry[field_key] is not None:
                    try:
                        values.append(float(entry[field_key]))
                        timestamp = datetime.strptime(entry['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                        days.append(timestamp.strftime('%a'))
                    except (ValueError, TypeError, KeyError):
                        continue
            
            return values, days
    except Exception as e:
        print(f"Error fetching ThingSpeak data: {e}")
    
    return None, None

def get_current_thingspeak_values():
    """Get the most recent values from ThingSpeak for all fields"""
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
    params = {
        'api_key': THINGSPEAK_READ_API_KEY,
        'timezone': 'Asia/Kolkata'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        current_values = {}
        for name, field in THINGSPEAK_FIELDS.items():
            field_key = f'field{field}'
            if field_key in data and data[field_key] is not None:
                try:
                    current_values[name] = float(data[field_key])
                except (ValueError, TypeError):
                    current_values[name] = 0
            else:
                current_values[name] = 0
        
        return current_values
    except Exception as e:
        print(f"Error fetching current ThingSpeak values: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        cursor.execute("SELECT name, password FROM user WHERE email = ? AND password = ?", (email, password))
        result = cursor.fetchone()
        connection.close()
        
        if result:
            session['user_name'] = result[0]
            return redirect(url_for('home'))
        else:
            flash('Sorry, Incorrect Credentials Provided, Try Again', 'error')
            return redirect(url_for('index'))
    
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        mobile = request.form.get('phone', '')
        email = request.form['email']
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        try:
            cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
            connection.commit()
            connection.close()
            flash('Successfully Registered', 'success')
            return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            connection.close()
            flash('Email already registered', 'error')
            return redirect(url_for('index', register='true'))
    
    return render_template('index.html')

@app.route('/home')
def home():
    if 'user_name' not in session:
        return redirect(url_for('index'))
    return render_template('home.html', username=session['user_name'])

@app.route('/dashboard')
def dashboard():
    if 'user_name' not in session:
        return redirect(url_for('home'))
    return render_template('home.html', username=session['user_name'])

@app.route('/learn')
def learn():
    if 'user_name' not in session:
        return redirect(url_for('index'))
    return render_template('learn.html', username=session['user_name'])

@app.route('/soil')
def soil():
    if 'user_name' not in session:
        return redirect(url_for('index'))
    
    # Get current sensor values from ThingSpeak
    current_values = get_current_thingspeak_values()
    
    # If no data from ThingSpeak, use default values
    if not current_values:
        current_values = {
            'Moisture': 62.3,
            'Temperature': 24.5,
            'Humidity': 65.0,
            'N-value': 35,
            'P-value': 25,
            'K-value': 30
        }
    
    # Get historical data for charts
    moisture_values, days = get_thingspeak_data(THINGSPEAK_FIELDS['Moisture'])
    temperature_values, _ = get_thingspeak_data(THINGSPEAK_FIELDS['Temperature'])
    humidity_values, _ = get_thingspeak_data(THINGSPEAK_FIELDS['Humidity'])
    n_values, _ = get_thingspeak_data(THINGSPEAK_FIELDS['N-value'])
    p_values, _ = get_thingspeak_data(THINGSPEAK_FIELDS['P-value'])
    k_values, _ = get_thingspeak_data(THINGSPEAK_FIELDS['K-value'])
    
    # If no historical data, use default values
    if not moisture_values:
        moisture_values = [58, 62, 65, 60, 63, 61, 62]
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    if not temperature_values:
        temperature_values = [24, 25, 26, 24, 23, 24, 25]
    
    if not humidity_values:
        humidity_values = [65, 64, 66, 63, 65, 64, 65]
    
    if not n_values:
        n_values = [35, 34, 36, 35, 34, 35, 36]
    
    if not p_values:
        p_values = [25, 24, 26, 25, 24, 25, 26]
    
    if not k_values:
        k_values = [30, 29, 31, 30, 29, 30, 31]
    
    # Determine nutrient level category
    nutrient_total = current_values.get('N-value', 0) + current_values.get('P-value', 0) + current_values.get('K-value', 0)
    if nutrient_total > 100:
        nutrient_level = "High"
    elif nutrient_total > 50:
        nutrient_level = "Medium"
    else:
        nutrient_level = "Low"
    
    return render_template('soil.html',
                         username=session['user_name'],
                         temperature=current_values.get('Temperature', 0),
                         moisture=current_values.get('Moisture', 0),
                         humidity=current_values.get('Humidity', 0),
                         nutrient_level=nutrient_level,
                         nitrogen=current_values.get('N-value', 0),
                         phosphorus=current_values.get('P-value', 0),
                         potassium=current_values.get('K-value', 0),
                         moisture_values=moisture_values,
                         temperature_values=temperature_values,
                         humidity_values=humidity_values,
                         n_values=n_values,
                         p_values=p_values,
                         k_values=k_values,
                         days=days)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if 'user_name' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('image.html', username=session['user_name'])
    
    if request.method == 'POST':
        # Clear previous images
        dirPath = "static/images"
        if os.path.exists(dirPath):
            fileList = os.listdir(dirPath)
            for fileName in fileList:
                os.remove(os.path.join(dirPath, fileName))
        else:
            os.makedirs(dirPath)
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('image'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('image'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            image = cv2.imread(filepath)
            
            # Generate processed images
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('static/gray.jpg', gray_image)
            
            edges = cv2.Canny(image, 100, 200)
            cv2.imwrite('static/edges.jpg', edges)
            
            _, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            cv2.imwrite('static/threshold.jpg', threshold2)
            
            kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel_sharpening)
            cv2.imwrite('static/sharpened.jpg', sharpened)
            
            # Load and process the model
            IMG_SIZE = 50
            LR = 1e-3
            MODEL_NAME = 'leafdisease-{}-{}.model'.format(LR, '2conv-basic')
            
            def process_verify_data():
                verifying_data = []
                img_num = os.path.splitext(filename)[0]
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                return verifying_data
            
            verify_data = process_verify_data()
            
            tf.compat.v1.reset_default_graph()
            convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
            
            # Model architecture
            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 64, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 128, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 64, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            
            convnet = fully_connected(convnet, 1024, activation='relu')
            convnet = dropout(convnet, 0.8)
            convnet = fully_connected(convnet, 19, activation='softmax')
            convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                               loss='categorical_crossentropy', name='targets')
            
            model = tflearn.DNN(convnet, tensorboard_dir='log')
            
            if os.path.exists(f"{MODEL_NAME}.meta"):
                model.load(MODEL_NAME)
                print('model loaded!')
            
            # Make prediction
            data = verify_data[0][0].reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])[0]
            
            class_idx = np.argmax(model_out)
            accuracy = f"{model_out[class_idx] * 100:.2f}%"
            
            label_dict = {
                0: 'Aphids_cotton_leaf',
                1: 'Army_worm_cotton_leaf',
                2: 'Bacterial_Blight',
                3: 'Healthy_cotton',
                4: 'Powdery_Mildew',
                5: 'Target_spot',
                6: 'Fussarium_wilt',
                7: 'Paddy_bacterial',
                8: 'Paddy_brownspot',
                9: 'Paddy_Leafsmut',
                10: 'Banana_cordana',
                11: 'Banana_healthy',
                12: 'Banana_pestalotiopsis',
                13: 'Banana_sigatoka',
                14: 'Tomato_bacterial_spot',
                15: 'Tomato_healthy',
                16: 'Tomato_leafmold',
                17: 'Tomato_spectoria',
                18: 'Tomato_yellow_curl_leaf'
            }
            
            remedy_dict = {
                'Aphids_cotton_leaf': [
                    "Methyl demeton 25 EC 500ml", "Dimethoate 30 EC 500ml", "Acetamiprid 20% SP 50 g",
                    "Azadirachtin 0.03% EC 500 ml", "Buprofezin 25% SC1000 ml", "Carbosulfan 25%DS 60g/kg of seed"
                ],
                'Army_worm_cotton_leaf': [
                    "Monitor the field, handpick diseased plants and bury them.",
                    "Use sticky yellow plastic traps.", "Spray organophosphates", 
                    "Use carbamates", "Use copper fungicides"
                ],
                'Bacterial_Blight': [
                    "Monitor the field, handpick diseased plants and bury them.",
                    "Use sticky yellow plastic traps.", "Spray organophosphates",
                    "Use carbamates", "Use copper fungicides"
                ],
                'Powdery_Mildew': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Target_spot': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Fussarium_wilt': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Paddy_bacterial': [
                    "Use disease-resistant varieties.", "Crop rotation.",
                    "Proper cultural practices.", "Chemical control"
                ],
                'Paddy_brownspot': [
                    "Crop rotation.", "Proper cultural practices.", 
                    "Fungicide application.", "Seed treatment."
                ],
                'Paddy_Leafsmut': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Banana_cordana': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Banana_pestalotiopsis': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Banana_sigatoka': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Tomato_bacterial_spot': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Tomato_leafmold': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ],
                'Tomato_spectoria': [
                    "Handpick and bury diseased plants.", "Use yellow plastic traps.",
                    "Spray insecticides (organophosphates)", "Use carbamates", "Use copper fungicides"
                ],
                'Tomato_yellow_curl_leaf': [
                    "Remove and destroy infected leaves.", "Use copper spray organically.",
                    "Use chemical fungicides like chlorothalonil"
                ]
            }
            
            str_label = label_dict[class_idx]
            remedies = remedy_dict.get(str_label, [])
            
            return render_template('results.html',
                                status=str_label,
                                accuracy=accuracy,
                                remedie="The remedies are:" if remedies else "",
                                remedie1=remedies,
                                ImageDisplay=f"/static/images/{filename}",
                                ImageDisplay1="/static/gray.jpg",
                                ImageDisplay2="/static/edges.jpg",
                                ImageDisplay3="/static/threshold.jpg",
                                ImageDisplay4="/static/sharpened.jpg",
                                username=session['user_name'])
        else:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
            return redirect(url_for('image'))

@app.route('/logout')
def logout():
    session.pop('user_name', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)