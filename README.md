Instruction for running AI
1. Make sure you have version 3.xx upwards of python
2. install Flask 'pip install flask, tensorflow, numpy'
3. run 'python app.py' to start server
4. you can have access to the api by calling POST: http://localhost:5000/predict_price with data: {
    "features": [59.99, 49.99, 500000, "High"]
}

for test purpose.