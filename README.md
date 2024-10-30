## Fashion Calendar Backend Readme

This project serves as the focal point that connects the machine learning models, the frontend, and the database. The API processes user input received from the frontend, interacts with machine learning models and metadata APIs, and returns the appropriate responses back to the frontend.

# Installation:
To set up the project locally, follow these steps:
1. Clone the repository:   
    - git clone https://github.com/your-repo/image-upload-recommendation.git
    - cd image-upload-recommendation
2. Create and activate a virtual environment:
   -  python3 -m venv venv
    - source venv/bin/activate  # On Windows, use: venv\Scripts\activate
3. Install the dependancies
   - pip install -r requirements.txt
4. Setup the Google Cloud SDK
     - gcloud auth application-default login
5. To run the application using google cloud services
      - gcloud app deploy
   To run the application on local host:
      - python app2.py

# Google Cloud Setup:
This project uses Google Cloud services for storage and hosting. You will need:

  - A Google Cloud Storage bucket for storing images, metadata, and utility matrices.
  - A Google Cloud service account key (service_account_key.json) with the appropriate permissions to access storage and run the application.

# Available Endpoints:
1. /
     Description: A simple home endpoint to confirm the API is running.
     Method: GET
2. /upload_image
     Description: Uploads an image and stores metadata returned by an external classification API.
     Method: POST
     Required:
          file: The image file to upload.
          user_id: The ID of the user uploading the image.
3. /get_images
     Description: Fetches all uploaded images for a user.
     Method: GET
     Required:
           user_id: The ID of the user.
4. /generate_recommendation
     Description: Generates a clothing recommendation based on uploaded images and metadata.
     Method: POST
     Required:
           user_id: The ID of the user.

# Usage
To use the API:
   Upload an image: Send a POST request to /upload_image with the image file and user ID.
   Fetch images: Send a GET request to /get_images with the user ID.
   Generate recommendations: Send a POST request to /generate_recommendation with the user ID.


# Detailed Function Descriptions
   1. upload_image
         This function handles the upload of images to Google Cloud Storage and calls an external API for metadata classification. The metadata is then saved as a JSON file in the cloud 
         and returned to the frontend
      
   2. get_images
          This function fetches all images uploaded by a user and returns signed URLs for accessing them. These URLs are passed back to the frontend.
      
   4. generate_recommendation
          This function processes metadata from uploaded images, communicates with external machine learning models, and returns personalized clothing recommendations. The utility matrix 
          for each user is updated based on how frequently each item is recommended.



