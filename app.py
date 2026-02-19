import sys
import os
import logging
import subprocess
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configure logging first with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the correct project root directory to Python path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(BASE_PATH, 'Safe-Path-Navigation-on-Lunar-terrains')
planning_project_path = os.path.join(project_root, 'planning_project')

# Print current directory structure for debugging
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"BASE_PATH: {BASE_PATH}")
logger.debug(f"Project root: {project_root}")
logger.debug(f"NPY_FOLDER path: {os.path.join(BASE_PATH, 'predicted_npy')}")

# Add both paths to sys.path
sys.path.append(project_root)
sys.path.append(os.path.dirname(project_root))

# Mock cupy with numpy before any other imports
import numpy as np
import types
cupy_mock = types.ModuleType('cupy')
cupy_mock.__dict__.update(np.__dict__)
sys.modules['cupy'] = cupy_mock
import builtins
builtins.cp = np

logger.info(f"Project root path: {project_root}")
logger.info(f"Python path: {sys.path}")

# Now import Flask and other dependencies
from flask import Flask, render_template, request, jsonify, send_file, session
# from utils.npy_generator import generate_npy
# from utils.height_map_generator import generate_height_map
from npy_prediction import predict_and_save_height_maps_npy
import matplotlib.pyplot as plt
import io
import base64
import traceback
from npy_prediction import ResNetUNet
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetUNet().to(device)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
NPY_FOLDER = 'predicted_npy'
input_dir = "C:/Planetary_Terrains"
output_dir = "C:/Planetary_Terrains/Planetary_Terrains/predicted_npy"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add secret key for session
app.secret_key = 'your_secret_key_here'  # Add this near the app initialization

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(NPY_FOLDER, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        process_type = request.form.get('type', 'heightmap')
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        # Log the file details
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Process type: {process_type}")

        # Clean the filename and create paths
        secure_filename = file.filename.replace(" ", "_")
        image_path = os.path.join(input_dir, secure_filename)
        npy_filename = os.path.splitext(secure_filename)[0] + '.npy'
        predicted_npy_file = os.path.join(output_dir, npy_filename)

        # Save uploaded image
        file.save(image_path)
        logger.info(f"✅ Image saved successfully to: {image_path}")

        # Generate predictions using the model
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load("C:/Planetary_Terrains/Planetary_Terrains/resnet_mode.pth", map_location=device))
            model.eval()
            predict_and_save_height_maps_npy(model, input_dir, output_dir, device)
            logger.info("✅ Model prediction completed successfully")
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to generate predictions'}), 500

        # Verify NPY file was created
        if not os.path.exists(predicted_npy_file):
            logger.error(f"❌ NPY file not found at: {predicted_npy_file}")
            logger.error(f"Directory contents: {os.listdir(output_dir)}")
            return jsonify({'error': 'Generated NPY file not found'}), 500
        else:
            logger.info(f"✅ NPY file found at: {predicted_npy_file}")

        if process_type == 'npy':
            try:
                return send_file(
                    predicted_npy_file,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name=npy_filename
                )
            except Exception as e:
                logger.error(f"Error sending NPY file: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': 'Failed to send NPY file'}), 500
        else:
            try:
                # Load and visualize the predicted height map
                data = np.load(predicted_npy_file, allow_pickle=True)
                height_map = data.item()['height']
                logger.info("✅ Height map loaded successfully")

                # Create figure
                plt.figure(figsize=(6, 6))
                plt.imshow(height_map, cmap="terrain")
                plt.colorbar(label="Height")
                plt.title("Height Map")
                plt.axis("off")

                # Convert plot to base64 string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()
                
                height_map_base64 = base64.b64encode(buf.getvalue()).decode()
                logger.info("✅ Height map visualization completed")

                return jsonify({
                    'message': 'Processing complete',
                    'height_map': height_map_base64
                })
            except Exception as e:
                logger.error(f"Error generating height map: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': 'Failed to generate height map'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Add a test endpoint to verify the server is working
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Server is running'}), 200

@app.route('/apply_path', methods=['POST'])
def apply_path():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Store parameters in session
        session['path_params'] = {
            'start_x': data.get('start_x'),
            'start_y': data.get('start_y'),
            'goal_x': data.get('goal_x'),
            'goal_y': data.get('goal_y')
        }

        # Start async task
        task_id = str(uuid.uuid4())
        session['task_id'] = task_id
        executor.submit(run_path_planning, task_id, session['path_params'], os.path.join(tempfile.gettempdir(), f'path_result_{task_id}.json'))

        return jsonify({
            'status': 'processing',
            'task_id': task_id
        })

    except Exception as e:
        logger.error(f"Error initiating path planning: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_path_status/<task_id>')
def check_path_status(task_id):
    try:
        result_file = os.path.join(tempfile.gettempdir(), f'path_result_{task_id}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            os.remove(result_file)  # Clean up
            return jsonify(result)
        return jsonify({'status': 'processing'}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_path_planning(task_id, params, result_file):
    try:
        script_path = os.path.join(project_root, 'path_planning.py')
        
        logger.info(f"Starting path planning for task {task_id}")
        logger.info(f"Using script path: {script_path}")
        logger.info(f"Parameters: {params}")

        # Ensure the script exists
        if not os.path.exists(script_path):
            raise Exception(f"Path planning script not found at: {script_path}")

        # Add the project root to PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Run the path planning script
        process = subprocess.run(
            [sys.executable, script_path,
             '--start_x', str(params['start_x']),
             '--start_y', str(params['start_y']),
             '--goal_x', str(params['goal_x']),
             '--goal_y', str(params['goal_y'])],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
            cwd=project_root  # Set working directory to project root
        )

        # Log the complete output for debugging
        logger.info("Path planning output:")
        logger.info(f"STDOUT: {process.stdout}")
        if process.stderr:
            logger.error(f"STDERR: {process.stderr}")

        if process.returncode != 0:
            error_msg = f"Path planning failed: {process.stderr}"
            logger.error(error_msg)
            with open(result_file, 'w') as f:
                json.dump({
                    'status': 'error',
                    'error': error_msg
                }, f)
            return

        result = {
            'status': 'completed',
            'visualization_path': None,
            'path_coords': None
        }

        # Parse output
        for line in process.stdout.split('\n'):
            if 'VISUALIZATION_PATH:' in line:
                result['visualization_path'] = line.split('VISUALIZATION_PATH:')[1].strip()
                logger.info(f"Found visualization path: {result['visualization_path']}")
            elif 'PATH_COORDS:' in line:
                try:
                    path_data = line.split('PATH_COORDS:')[1].strip()
                    result['path_coords'] = json.loads(path_data)
                    logger.info(f"Found path coordinates")
                except Exception as e:
                    logger.error(f"Failed to parse path coordinates: {e}")

        # Save results
        with open(result_file, 'w') as f:
            json.dump(result, f)

    except Exception as e:
        error_msg = f"Error in path planning: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        with open(result_file, 'w') as f:
            json.dump({
                'status': 'error',
                'error': error_msg,
                'details': traceback.format_exc()
            }, f)

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

if __name__ == '__main__':
    # Print initial debug information
    logger.info("Starting Flask application...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Matplotlib version: {matplotlib.__version__}")
    
    app.run(debug=True,port=5003)
