# Boiler Jet Jinx

A Flask-based web application for flight delay prediction and optimization.

Live Website : https://jetjinx.onrender.com/ 

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

## Large Files and Data

Some large files are not included in this repository due to size limitations. You can download them from Google Drive:

[Download Large Files from Google Drive](https://drive.google.com/drive/folders/1guk7lV7rDZ8indb9tNzlbUZrZBG2OW9m?usp=drive_link)

The following files are available in the Google Drive folder:
- `flights.csv` (565MB) - Main dataset for flight delays
- `ProjectAirline.ipynb` (1.1MB) - Jupyter notebook with analysis
- Various PNG files for visualizations

After downloading these files, place them in the root directory of the project to ensure all code works as expected.

## License

MIT License 
