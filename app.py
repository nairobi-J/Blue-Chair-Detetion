# app.py
#import model_inference
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from model_inference import BlueChairDetector # Import your model inference class
import io

app = Flask(__name__)

# Configuration for file uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load your model once when the Flask app starts
# This avoids reloading the model for every single request, which is very inefficient.
print("Initializing BlueChairDetector...")
try:
    detector = BlueChairDetector(model_path='model/best.pt')
    if detector.model is None:
        print("CRITICAL ERROR: Model failed to load. The application will not function correctly.")
        # Optionally, you could raise an exception here or set a flag to prevent requests
except Exception as e:
    print(f"CRITICAL ERROR during detector initialization: {e}")
    # Handle this more gracefully in production, e.g., show a maintenance page.
    detector = None # Ensure detector is None if initialization fails

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_blue_chair():
    if detector is None:
        return jsonify({'error': 'Model not initialized. Server error.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create a unique filename to prevent clashes and issues with caching
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{os.urandom(8).hex()}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath) # Save the uploaded file temporarily

        with open(filepath, 'rb') as f:
            image_bytes = f.read()

        # Perform blue chair detection
        processed_image_bytes, detections_info = detector.detect_blue_chairs(image_bytes)

        # Clean up the uploaded file immediately after reading
        os.remove(filepath)

        if processed_image_bytes is None:
            return jsonify({'error': detections_info}), 500 # detections_info will contain the error message

        # Save the processed image for serving back to the user
        result_filename = f"detected_{unique_filename}"
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        with open(result_filepath, 'wb') as f:
            f.write(processed_image_bytes)
        
        # Prepare the JSON response
        response = {
            'message': 'Detection successful',
            'original_filename': filename,
            'result_image_url': f'/results/{result_filename}',
            'detections': detections_info 
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif.'}), 400

# Route to serve the processed images
@app.route('/results/<filename>')
def serve_result_image(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), mimetype='image/jpeg')

if __name__ == '__main__':
    # For development, debug=True is useful. For production, set debug=False.
    # host='0.0.0.0' makes the server accessible from other devices on your network.
    app.run(debug=True, host='0.0.0.0')