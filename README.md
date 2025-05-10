# Breast Cancer Histopathological Image Classifier

A deep learning web application that classifies breast cancer histopathological images as benign or malignant using a fine-tuned ResNet50 model. This project leverages transfer learning on the BreakHis dataset to provide accurate classification of microscopic breast tissue images.


## Preview
![image](https://github.com/user-attachments/assets/c45271d8-1075-4884-97eb-84413a59ba5e)



## Tech Stack

- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: React.js, CSS3
- **Model**: Fine-tuned ResNet50 CNN

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+ with pip
- Node.js 14+ with npm


### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jayeshtripathi/breast-cancer-classifier.git
cd breast-cancer-classifier
```

2. **Set up the backend**
```bash
# Create and activate a virtual environment
python -m venv env_name
source env_name/bin/activate

# Install python dependencies
pip install -r requirements.txt

# Obtain resnet50_breakhis_model.json and resnet50_breakhis_weights.weights.h5 by executing the cells in finetune_resnet50.ipynb. This was done with Kaggle notebook, but the same can be executed in local machine by downloading the BreakHis dataset locally.
# Place resnet50_breakhis_model.json and resnet50_breakhis_weights.weights.h5 in the root directory
```

3. **Set up the frontend**
```bash
cd frontend
npm install
```


### Running the Application

1. **Start the Flask backend**
```bash
# From the root directory
python app.py
```

2. **Start the React frontend**
```bash
# In a new terminal, from the frontend directory
npm start
```

3. **Access the application**

Open your browser and navigate to [http://localhost:3000](http://localhost:3000)

## Usage

1. Toggle between dark and light mode using the button in the top-right corner
2. Click "Select Image" to upload a histopathological image
3. Click "Classify Image" to process the image
4. View the classification result (Benign or Malignant) and confidence score

## Model Information

The classifier uses a ResNet50 convolutional neural network fine-tuned on the BreakHis dataset containing 7,909 microscopic images of breast tumor tissue. The model has been specifically trained to recognize patterns in histopathological images to classify breast tissue samples as benign or malignant.

### Dataset

The [BreakHis dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) contains 7,909 microscopic images of breast tumor tissue collected from 82 patients using different magnification factors (40X, 100X, 200X, and 400X).

