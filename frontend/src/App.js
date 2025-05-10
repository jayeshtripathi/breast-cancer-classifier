import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : false;
  });

  useEffect(() => {
    document.body.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select an image to upload.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'An error occurred during prediction.');
      }
    } catch (err) {
      setError('Failed to connect to the server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="toggle-container">
        <button
          className="toggle-btn"
          onClick={() => setDarkMode(!darkMode)}
          aria-label="Toggle dark mode"
        >
          {darkMode ? '☀️' : '🌙'}
        </button>
      </div>
      <header className="App-header">
        <h1 className="title">
          Breast Cancer
          <br />
          <span className="subtitle">Histopathological Image Classifier</span>
        </h1>
        
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="button-group">
            <label className="file-input-label">
              Select Image
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? 'Processing...' : 'Classify Image'}
            </button>
          </div>
        </form>
        
        <div className="model-info">
          <h3>About the Model</h3>
          <p>This classifier uses a pre-trained ResNet50 convolutional neural network to analyze histopathological images of breast tissue and determine whether they show benign or malignant characteristics. The model was trained on the BreakHis dataset containing 7,909 microscopic images of breast tumor tissue.</p>
        </div>
        
        {(file || result || error) && (
          <div className="content-container">
            {file && (
              <div className="image-preview">
                <h3>Input Image:</h3>
                <img
                  src={URL.createObjectURL(file)}
                  alt="Input"
                  style={{ maxWidth: '300px', maxHeight: '300px', borderRadius: '8px' }}
                />
              </div>
            )}
            <div className="result-error-container">
              {result && (
                <div className="result">
                  <h3>Result:</h3>
                  <p>Classification: {result.class_label}</p>
                  <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
                </div>
              )}
              {error && (
                <div className="error">
                  <h3>Error:</h3>
                  <p>{error}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
