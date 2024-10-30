import os
import json
import asyncio
import aiohttp
from flask import Flask, request, jsonify
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta
from flask_cors import CORS
import requests as hugging_face_request
import ast
import requests
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_FILE = 'service_account_key.json'

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

storage_client = storage.Client(credentials=credentials)
bucket_name = 'cohesive-vine-437901-v1.appspot.com'


@app.route('/')
def home():
    return "Welcome to the Image Upload and Recommendation API! Use /upload_image, /get_images, /generate_recommendation."


async def upload_metadata(file_path, user_id):
    async with aiohttp.ClientSession() as session:
        url = "https://bewajafarwah--hf-stackup-image-classifier-fastapi-app.modal.run/metadata"

        for attempt in range(2):
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': f}
                    async with session.get(url, params=files, timeout=30) as response:
                        if response.status != 200:
                            raise Exception(f"Error from API: {response.status}, {await response.text()}")

                        response_data = await response.json()

                print(f"Metadata response for user {user_id}: {response_data}")

                json_blob_path = f"user_{user_id}_metadata/{os.path.basename(file_path).rsplit('.', 1)[0]}.json"
                bucket = storage_client.get_bucket(bucket_name)
                blob = bucket.blob(json_blob_path)
                blob.upload_from_string(json.dumps(response_data), content_type='application/json')

                return response_data
            except Exception as e:
                print(f"Error on attempt {attempt + 1} for user {user_id}: {e}")
                if attempt == 1:
                    return None

    return None


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file in the request"}), 400

    file = request.files['file']
    user_id = request.form.get('user_id')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    blob_path = f"user_{user_id}/{file.filename}"

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_file(file)

    url = "https://bewajafarwah--hf-stackup-image-classifier-fastapi-app.modal.run/metadata"

    try:
        file.stream.seek(0)
        files = [
            ('file', (file.filename, file, 'image/png'))
        ]
        response = requests.get(url, files=files, timeout=30)
        response.raise_for_status()
        metadata_response = response.json()

        metadata = metadata_response.get("metadata", {})
        metadata["image_name"] = file.filename.rsplit('.', 1)[0]

        json_blob_name = f"user_{user_id}_metadata/{file.filename.rsplit('.', 1)[0]}.json"
        json_blob = bucket.blob(json_blob_name)

        json_blob.upload_from_string(json.dumps(metadata), content_type='application/json')

    except Exception as e:
        logger.error(f"Error calling metadata API: {e}")
        metadata_response = None

    return jsonify({
        "message": "File uploaded successfully!",
        "file_url": f"gs://{bucket_name}/{blob_path}",
        "metadata_response": metadata
    }), 201





@app.route('/get_images', methods=['GET'])
def get_images():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        bucket = storage_client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=f"user_{user_id}/"))

        if not blobs:
            return jsonify({
                "user_id": user_id,
                "message": "No uploads yet. Please upload an image."
            }), 200

        image_urls = []
        for blob in blobs:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=1),
                method="GET"
            )
            image_urls.append(signed_url)

        return jsonify({
            "user_id": user_id,
            "images": image_urls
        })

    except Exception as e:
        print(f"Error fetching images for user {user_id}: {e}")

        return jsonify({
            "error": "An error occurred while fetching images",
            "details": str(e)
        }), 500


@app.route('/generate_recommendation', methods=['POST'])
def generate_recommendation():
    data = request.get_json()
    user_id = data.get("user_id")

    if user_id is None:
        return jsonify({"error": "User ID is required"}), 400

    bucket = storage_client.get_bucket(bucket_name)

    utility_matrix = {}
    utility_file_path = f"user_{user_id}_utility/utility_matrix.json"
    try:
        utility_blob = bucket.blob(utility_file_path)
        if utility_blob.exists():
            utility_matrix = json.loads(utility_blob.download_as_string().decode('utf-8'))
            logger.info(f"Loaded existing utility matrix: {utility_matrix}")
        else:
            logger.info(f"No existing utility matrix found for user {user_id}, initializing fresh matrix.")
    except Exception as e:
        logger.error(f"Error loading utility matrix for user {user_id}: {e}")
        utility_matrix = {}

    blobs = list(bucket.list_blobs(prefix=f"user_{user_id}_metadata/"))

    if not blobs:
        return jsonify({
            "user_id": user_id,
            "message": "No uploads yet. Please upload an image."
        }), 200

    metadata_urls = []
    for blob in blobs:
        str_blob = blob.download_as_string().decode('utf-8')
        if len(str_blob) != 0:
            dict_blob = ast.literal_eval(str_blob)
            metadata_urls.append(dict_blob)

    myobj2 = {"data": [metadata_urls]}

    response = hugging_face_request.post(url="https://abhi995-generatelocalmatrix.hf.space/gradio_api/call/predict", json=myobj2)
    response_code = ast.literal_eval(response.content.decode('utf-8')).get('event_id')

    url_2 = f"https://abhi995-generatelocalmatrix.hf.space/gradio_api/call/predict/{response_code}"
    respose_local_matrix = hugging_face_request.get(url=url_2)
    rlm_txt = respose_local_matrix.content.decode('utf-8')
    rlm_txt = json.loads(rlm_txt[rlm_txt.index("["):].strip())

    image_order = rlm_txt[0].get("image_order")
    local_matrix = rlm_txt[0].get("local_matrix")
    local_matrix = [[-1000 if val is None else val for val in row] for row in local_matrix]

    min_utility = 1 / 256
    for image in image_order:
        if image not in utility_matrix:
            utility_matrix[image] = 1

    final_data = {
        "data": [
            {
                "local_matrix": local_matrix,
                "utility": [utility_matrix[img] for img in image_order],
                "num_days": 5,
                "image_order": image_order
            }]
    }

    response_recc_1 = hugging_face_request.post(url="https://abhi995-getrecommendation.hf.space/gradio_api/call/predict", json=final_data)
    response_recc_code = ast.literal_eval(response_recc_1.content.decode('utf-8')).get('event_id')

    url_3 = f"https://abhi995-getrecommendation.hf.space/gradio_api/call/predict/{response_recc_code}"
    response_getrecc = hugging_face_request.get(url=url_3)
    getrecc_txt = response_getrecc.content.decode('utf-8')
    getrecc_txt = json.loads(getrecc_txt[getrecc_txt.index("["):].strip())
    intermediary_recc = getrecc_txt[0].get("result", [])
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    final_recommendation = []

    image_extensions = ['.jpg', '.jpeg', '.png']

    for i, pair in enumerate(intermediary_recc):
        day = days_of_week[i % len(days_of_week)]

        image_urls = []
        for item in pair:
            found_url = None
            for ext in image_extensions:
                image_blob = bucket.blob(f"user_{user_id}/{item}{ext}")
                if image_blob.exists():
                    found_url = image_blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(hours=1),
                        method="GET"
                    )
                    break

            if found_url:
                image_urls.append(found_url)
            else:
                image_urls.append(None)

        final_recommendation.append({
            "day": day,
            "image_urls": image_urls
        })

        for item in pair:
            if item in utility_matrix:
                utility_matrix[item] /= 2
                if utility_matrix[item] < min_utility:
                    utility_matrix[item] = 1

    updated_utility_matrix = json.dumps(utility_matrix)
    utility_blob.upload_from_string(updated_utility_matrix, content_type='application/json')
    logger.info(f"Saved updated utility matrix for user {user_id}: {utility_matrix}")

    return jsonify({
        "final_recc": final_recommendation,
        "updated_utility_matrix": [utility_matrix[img] for img in image_order]
    })


if __name__ == '__main__':
    app.run(debug=True, threaded = false)
