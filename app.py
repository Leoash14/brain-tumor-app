from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import joblib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the SVM model and label map from saved file
data = joblib.load('model.pkl')  # or 'svm_model.pkl' if that's your filename
model = data['model']
label_map = data['label_map']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match training
    image_array = np.array(image)

    if len(image_array.shape) == 2:  # Grayscale to 3 channels
        image_array = np.stack((image_array,) * 3, axis=-1)

    if image_array.shape[-1] == 4:  # If image has alpha channel, remove it
        image_array = image_array[:, :, :3]

    image_array = image_array.astype('float32')
    image_array = np.expand_dims(image_array, axis=0)

    # Use ResNet50 preprocess_input
    from tensorflow.keras.applications.resnet50 import preprocess_input
    image_array = preprocess_input(image_array)

    # Load ResNet50 feature extractor
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    features = feature_extractor.predict(image_array)
    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    result = None
    error = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = filepath

            try:
                image = Image.open(filepath)
                img = preprocess_image(image)

                # ✅ Use predict_proba to get confidence
                proba = model.predict_proba(img)[0]
                pred = np.argmax(proba)
                confidence = round(proba[pred] * 100, 2)

                prediction = label_map.get(pred, "Unknown")
                result = f"{prediction} (Confidence: {confidence}%)"

            except Exception as e:
                error = f"Error processing image: {e}"

    return render_template('index.html', result=result,prediction=prediction, image_path=image_path, error=error,confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
