from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf 
from flask import Flask, render_template, request,session,logging,url_for,redirect,flash
import webcolors 
import mysql.connector
import os
 
from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import webcolors
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model 


from werkzeug.utils import secure_filename
app = Flask(__name__)
 
app.secret_key=os.urandom(24)
app.static_folder = 'static'

# Upload folder path
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load the chatbot
chatbot = ChatBot(
    'CARS-CHAT BOT',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)

# Training chatbot with personal Q&A data
training_data_quesans = open('C:/Users/Rishi/OneDrive/Desktop/carbot/car/cars.txt').read().splitlines()
training_data = training_data_quesans

trainer = ListTrainer(chatbot)
trainer.train(training_data)

trainer_corpus = ChatterBotCorpusTrainer(chatbot) 
model = tf.keras.models.load_model("carmodel.h5") 
 

app.config['SECRET_KEY'] = 'sdsdsdsdsdsdsd'
 
#database connectivity
conn=mysql.connector.connect(host='localhost',port='3306',user='root',password='root',database='register')
cur=conn.cursor()
# Set upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
#model_path = os.path.join(os.getcwd(), 'model/model.pkl')
#with open(model_path, 'rb') as model_file:
    #model = pickle.load(model_file)

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/index")
def home():
    if 'id' in session:
        return render_template('index.html')
    else:
        return redirect('/')

# Route to handle chatbot text input
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText:  # Check if text is given
        response = str(chatbot.get_response(userText))
        return jsonify({"response": response})
    else:
        return jsonify({"error": "Error in processing"}), 400


@app.route('/')
def login():
    return render_template("login.html")

@app.route('/register')
def about():
    return render_template('register.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/login_validation',methods=['POST'])
def login_validation():
    email=request.form.get('email')
    password=request.form.get('password')

    cur.execute("""SELECT * FROM `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email,password))
    users = cur.fetchall()
    if len(users)>0:
        session['id']=users[0][0]
        flash('You were successfully logged in')
        return redirect('/index')
    else:
        flash('Invalid credentials !!!')
        return redirect('/')
    # return "The Email is {} and the Password is {}".format(email,password)
    # return render_template('register.html')

@app.route('/add_user',methods=['POST'])
def add_user():
    name=request.form.get('name') 
    email=request.form.get('uemail')
    password=request.form.get('upassword')

    #cur.execute("UPDATE users SET password='{}'WHERE name = '{}'".format(password, name))
    cur.execute("""INSERT INTO  users(name,email,password) VALUES('{}','{}','{}')""".format(name,email,password))
    conn.commit()
    cur.execute("""SELECT * FROM `users` WHERE `email` LIKE '{}'""".format(email))
    myuser=cur.fetchall()
    flash('You have successfully registered!')
    session['id']=myuser[0][0]
    return render_template("login.html")


# Car class labels
class_labels = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']

# Car models and descriptions dictionary
car_models_dict = {
    "Audi": {
        "models": ["Audi A4", "Audi Q7", "Audi A6", "Audi Q5"],
        "description": "Audi is a luxury brand known for high-performance sedans and SUVs with advanced features and refined interiors."
    },
    "Hyundai Creta": {
        "models": ["Creta SX", "Creta SX(O)", "Creta E", "Creta EX"],
        "description": "Hyundai Creta is a compact SUV known for its stylish design, fuel efficiency, and advanced safety features."
    },
    "Mahindra Scorpio": {
        "models": ["Scorpio S3+", "Scorpio S5", "Scorpio S7", "Scorpio S11"],
        "description": "Mahindra Scorpio is a powerful SUV with rugged build quality, suitable for off-road and urban environments."
    },
    "Rolls Royce": {
        "models": ["Rolls Royce Phantom", "Rolls Royce Ghost", "Rolls Royce Cullinan", "Rolls Royce Wraith"],
        "description": "Rolls Royce is a symbol of ultimate luxury and craftsmanship, offering bespoke vehicles for the elite."
    },
    "Swift": {
        "models": ["Swift LXi", "Swift VXi", "Swift ZXi", "Swift ZXi+ AGS"],
        "description": "Maruti Suzuki Swift is a popular hatchback known for its fuel efficiency, modern features, and affordability."
    },
    "Tata Safari": {
        "models": ["Safari XE", "Safari XM", "Safari XT+", "Safari XZ+"],
        "description": "Tata Safari is a rugged SUV offering powerful performance, spacious interiors, and advanced technology."
    },
    "Toyota Innova": {
        "models": ["Innova Crysta G", "Innova Crysta GX", "Innova Crysta VX", "Innova Crysta ZX"],
        "description": "Toyota Innova is a reliable MPV with excellent space, safety features, and strong performance."
    }
}


# Get dominant color of the car
def get_dominant_color(image_path, k=3):
    """Get dominant color of cropped car image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped_image = image.reshape((-1, 3))

    # K-Means clustering to get dominant color
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_image)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    color_name = get_color_name(dominant_color)

    return dominant_color, color_name


# Get color name from RGB values
def get_color_name(rgb_color):
    """Convert RGB to closest color name."""
    try:
        closest_name = webcolors.rgb_to_name(tuple(rgb_color))
    except ValueError:
        closest_name = closest_color(rgb_color)
    return closest_name


# Find the closest color in the CSS color list
def closest_color(requested_color):
    """Find the closest color if exact match is not available."""
    min_colors = {}
    for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(hex_code)
        distance = np.linalg.norm(np.array((r, g, b)) - np.array(requested_color))
        min_colors[distance] = name
    return min_colors[min(min_colors.keys())]



# Route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the image
    image = Image.open(filepath)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the car model
    pred_probs = model.predict(image)[0]
    pred_label = class_labels[np.argmax(pred_probs)]
    car_models = car_models_dict.get(pred_label, {}).get("models", ["No models available"])
    description = car_models_dict.get(pred_label, {}).get("description", "No description available.")

    result = f"Car Model Detected: {pred_label}"
    response_data = {
        "result": result,
        "models": car_models,
        "description": description,
        "image_url": url_for('static', filename=f"uploads/{filename}")
    }
    return jsonify(response_data), 200


# Handle combined request (text + image)
@app.route("/process", methods=["POST"])
def process():
    userText = request.form.get('msg')
    file = request.files.get("image")

    if userText and not file:
        # Process text only
        response = str(chatbot.get_response(userText))
        return jsonify({"response": response})

    elif file and not userText:
        # Process image only
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        pred_probs = model.predict(image)[0]
        pred_label = class_labels[np.argmax(pred_probs)]
        car_models = car_models_dict.get(pred_label, {}).get("models", ["No models available"])
        description = car_models_dict.get(pred_label, {}).get("description", "No description available.")

        result = f"Car Model Detected: {pred_label}"
        response_data = {
            "result": result,
            "models": car_models,
            "description": description,
            "image_url": url_for('static', filename=f"uploads/{filename}")
        }
        return jsonify(response_data), 200

    elif userText and file:
        # Process both text and image
        text_response = str(chatbot.get_response(userText))
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        pred_probs = model.predict(image)[0]
        pred_label = class_labels[np.argmax(pred_probs)]
        car_models = car_models_dict.get(pred_label, {}).get("models", ["No models available"])
        description = car_models_dict.get(pred_label, {}).get("description", "No description available.")

        result = f"Car Model Detected: {pred_label}"
        response_data = {
            "text_response": text_response,
            "result": result,
            "models": car_models,
            "description": description,
            "image_url": url_for('static', filename=f"uploads/{filename}")
        }
        return jsonify(response_data), 200

    else:
        return jsonify({"error": "Error in processing"}), 400
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_color):
    try:
        return webcolors.rgb_to_name(rgb_color)
    except ValueError:
        return closest_color(rgb_color)

# Get pixel color and name on mouse click
@app.route("/get_color", methods=["POST"])
def get_color():
    data = request.get_json()
    x, y, image_url = int(data["x"]), int(data["y"]), data["image_url"]

    # Map image path correctly
    image_path = image_url.replace("/static/", "static/")
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404

    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Error loading image"}), 500

    # Get color at the selected point (BGR format in OpenCV)
    b, g, r = image[y, x]
    color_hex = f"#{r:02x}{g:02x}{b:02x}"

    # Convert RGB to color name
    color_name = get_color_name((r, g, b))

    return jsonify({"color": color_hex, "color_name": color_name})

@app.route('/add_user',methods=['POST'])
def register():
    if recaptcha.verify():
        flash('New User Added Successfully')
        return redirect('/register')
    else:
        flash('Error Recaptcha') 
        return redirect('/register')

# Load car color model
model1 = load_model('carcolormodel.h5')
class_names = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']
color_hex_map = {
    'beige': '#F5F5DC', 'black': '#000000', 'blue': '#0000FF', 'brown': '#8B4513', 'gold': '#FFD700',
    'green': '#008000', 'grey': '#808080', 'orange': '#FFA500', 'pink': '#FFC0CB', 'purple': '#800080',
    'red': '#FF0000', 'silver': '#C0C0C0', 'tan': '#D2B48C', 'white': '#FFFFFF', 'yellow': '#FFFF00'
}

@app.route("/predict_color", methods=["POST"])
def predict_color():
    if "color_image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["color_image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess image
    image = Image.open(filepath).resize((224, 224)).convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model1.predict(img_array)
    predicted_index = np.argmax(predictions)
    color_name = class_names[predicted_index]
    color_hex = color_hex_map.get(color_name, '#000000')

    return jsonify({
        "color_name": color_name,
        "color_hex": color_hex
    })

@app.route('/logout')
def logout():
    session.pop('id')
    return redirect('/')
    
if __name__ == "__main__":
    app.run(debug=True)
