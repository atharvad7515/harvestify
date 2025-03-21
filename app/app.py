# Importing essential libraries and modules
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM



from flask import Flask, request, render_template
import logging
import re



from flask import Flask, render_template, request
from markupsafe import Markup  # Import Markup separately
import pickle
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9










model = pickle.load(open('modelcopy.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))



# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# url


# Loading crop recommendation model
# Updated to load the model saved with the new feature sequence: N, P, K, Temperature, Humidity, pH, Rainfall
with open("model.pkl", "rb") as file:
    crop_recommendation_model = pickle.load(file)

# =========================================================================================

# Custom functions for calculations

def weather_fetch(city_name):
    """
    Fetches and returns the temperature and humidity of a city.
    (No longer used in crop recommendation as temperature and humidity are now provided by the user.)
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label.
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

def get_price_data(state=None, district=None, commodity=None):
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
        "format": "json",
        "offset": 0,
        "limit": 10
    }
    
    # Add filters
    if state:
        params["filters[state]"] = state
    if district:
        params["filters[district]"] = district
    if commodity:
        params["filters[commodity]"] = commodity
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching data: {e}")
        return []

    record = {
    'state': item.get('state', 'N/A'),
    'district': item.get('district', 'N/A'),
    'commodity': item.get('commodity', 'N/A'),
    'min_price': item.get('min_price', 'N/A'),
    'max_price': item.get('max_price', 'N/A'),
    'market': item.get('market', 'N/A'),
    'arrival_date': item.get('arrival_date', 'N/A')
     }
    
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

def get_price_data(state=None, district=None, commodity=None):
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
        "format": "json",
        "offset": 0,
        "limit": 10
    }
    
    # Add filters
    if state:
        params["filters[state]"] = state
    if district:
        params["filters[district]"] = district
    if commodity:
        params["filters[commodity]"] = commodity

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # ✅ Initialize records list
        records = []

        # ✅ Ensure "records" exists in response
        if "records" in data and isinstance(data["records"], list):
            for item in data["records"]:  # ✅ Iterate over fetched records
                record = {
                    'state': item.get('state', 'N/A'),
                    'district': item.get('district', 'N/A'),
                    'commodity': item.get('commodity', 'N/A'),
                    'min_price': item.get('min_price', 'N/A'),
                    'max_price': item.get('max_price', 'N/A'),
                    'market': item.get('market', 'N/A'),
                    'arrival_date': item.get('arrival_date', 'N/A')
                }
                records.append(record)

        return records  # ✅ Now, 'records' is always defined

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching data: {e}")
        return []  # ✅ Always return an empty list on failure

@app.route('/api')
def market():
    state = request.args.get('state', '')
    district = request.args.get('district', '')
    commodity = request.args.get('commodity', '')
    
    price_data = get_price_data(state, district, commodity)
    return render_template('api.html', 
                           price_data=price_data,
                           state=state,
                           district=district,
                           commodity=commodity)



# @app.route('/api')
# def index():
#     state = request.args.get('state', '')
#     district = request.args.get('district', '')
#     commodity = request.args.get('commodity', '')
    
#     price_data = get_price_data(state, district, commodity)
#     return render_template('api.html', 
#                          price_data=price_data,
#                          state=state,
#                          district=district,
#                          commodity=commodity)

# API Endpoint
API_URL = "https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f"
API_KEY = "579b464db66ec23bdd00000114421da1118d49df5bdc6532b5fe6eb5"

def fetch_data():
    """Fetches data from the API."""
    params = {
        "api-key": API_KEY,
        "format": "json"
    }
    response = requests.get(API_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

@app.route("/govSchemes", methods=["GET", "POST"])
def govSchemes():
    data = fetch_data()
    
    if not data:
        return "Failed to fetch data from API", 500

    # Extract unique values for dropdowns
    records = data.get("records", [])
    states = sorted(set(record.get("StateName", "") for record in records))
    districts = sorted(set(record.get("DistrictName", "") for record in records))
    query_types = sorted(set(record.get("QueryType", "") for record in records))

    kcc_answer = None

    if request.method == "POST":
        selected_state = request.form.get("state")
        selected_district = request.form.get("district")
        selected_query_type = request.form.get("query_type")

        # Filter records
        for record in records:
            if (record.get("StateName") == selected_state and
                record.get("DistrictName") == selected_district and
                record.get("QueryType") == selected_query_type):
                kcc_answer = record.get("KccAns", "No Answer Found")
                break

    return render_template("govSchemes.html", states=states, districts=districts, query_types=query_types, kcc_answer=kcc_answer)


@app.route('/marathi')
def marathi_page():
    return render_template('marathi.html')







# Render home page
@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# Render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# Render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# Render disease prediction input page

# ===============================================================================================
# RENDER PREDICTION PAGES

# Render crop recommendation result page
# @app.route('/crop-predict', methods=['POST'])
# def crop_prediction():
#     title = 'Harvestify - Crop Recommendation'
#     if request.method == 'POST':
#         # Get input values from the form (ensure your form has these fields)
#         N = int(request.form['nitrogen'])
#         P = int(request.form['phosphorous'])
#         K = int(request.form['pottasium'])
#         temperature = float(request.form['temperature'])
#         humidity = float(request.form['humidity'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
        
#         # Prepare the data array in the order: N, P, K, Temperature, Humidity, pH, Rainfall
#         data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
#         my_prediction = crop_recommendation_model.predict(data)
#         final_prediction = my_prediction[0]

#         return render_template('crop-result.html', prediction=final_prediction, title=title)




# crop- Detection
@app.route('/cropDetection')
def cropDetection():
    return render_template("cropDetection.html")

@app.route("/cropDetection",methods=['POST'])
def Predict():
    N = request.form['Nitrogen']
    P = request.form['Phosphorus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('cropDetection.html',result = result)


# Render fertilizer recommendation result page
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        key = 'NHigh' if n < 0 else "Nlow"
    elif max_value == "P":
        key = 'PHigh' if p < 0 else "Plow"
    else:
        key = 'KHigh' if k < 0 else "Klow"

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)

# Render disease prediction result page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print("Error in disease prediction:", e)
    return render_template('disease.html', title=title)

@app.route('/weather')
def weather_view():
    return render_template('weather.html') 


@app.route('/location')
def location_view():
    return render_template('location.html')


# here ends


def format_output(text):
    """Convert Markdown bold syntax to HTML strong tags."""
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

# Define chatbot initialization
def initialise_llama3():
    try:    
        # Create chatbot prompt
        create_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are my personal assistant"),
                ("user", "Question: {question}")
            ]
        )

        # Initialize OpenAI LLM and output parser
        lamma_model = Ollama(model="llama3.2")
        format_output = StrOutputParser()

        # Create chain
        chatbot_pipeline = create_prompt | lamma_model | format_output
        return chatbot_pipeline
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise

# Initialize chatbot
chatbot_pipeline = initialise_llama3()

# Define route for home page
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    query_input = None
    output = None
    if request.method == 'POST':
        query_input = request.form.get('query-input')
        if query_input:
            try:
                response = chatbot_pipeline.invoke({'question': query_input})
                output = format_output(response)
            except Exception as e:
                logging.error(f"Error during chatbot invocation: {e}")
                output = "Sorry, an error occurred while processing your request."
    return render_template('chatbot.html', query_input=query_input, output=output)

# @app.route('/chatbot', methods=['GET', 'POST'])
# def main():
#     query_input = None
#     output = None
#     if request.method == 'POST':
#         query_input = request.form.get('query-input')
#         if query_input:
#             try:
#                 response = chatbot_pipeline.invoke({'question': query_input})
#                 output = format_output(response)
#             except Exception as e:
#                 logging.error(f"Error during chatbot invocation: {e}")
#                 output = "Sorry, an error occurred while processing your request."
#     return render_template('chatbot.html', query_input=query_input, output=output)
# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
