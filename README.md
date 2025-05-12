# Boiler Jet Jinx

A Flask-based web application for flight delay prediction and optimization.

## Features

- Flight delay prediction using machine learning
- Interactive data visualizations
- Optimization recommendations
- Modern, responsive UI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/AryanSaxena05/JetJinx.git
cd JetJinx
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5050`

## Project Structure

```
JetJinx/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/            # Static files (CSS, JS)
├── templates/         # HTML templates
├── models/           # ML models and preprocessors
└── utils/            # Utility functions
```

## Dependencies

- Flask 2.0.1
- pandas 1.3.3
- numpy 1.21.2
- scikit-learn 0.24.2
- plotly 5.3.1

## License

MIT License 