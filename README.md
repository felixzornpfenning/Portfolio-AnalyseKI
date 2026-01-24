# Portfolio-AnalyseKI
In diesen Programm können sie Aktienportfolien anlegen und diese mit dem S&P 500 vergleichen lassen.

## Setup Instructions

### 1. Install Python
- Download Python from https://www.python.org/downloads/ or install from Microsoft Store
- **Important**: Make sure to check "Add Python to PATH" during installation

### 2. Setup Virtual Environment
Run the setup script:
```bash
setup_venv.bat
```

Or manually:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
# Make sure virtual environment is activated
venv\Scripts\activate

# Run the portfolio analysis
python "Portfolio Analyse.py"
```

### 4. Deactivate Environment
```bash
deactivate
```

## Project Structure
```
Portfolio-AnalyseKI/
├── Portfolio Analyse.py    # Main analysis script
├── requirements.txt        # Python dependencies
├── setup_venv.bat         # Automated setup script
├── venv/                  # Virtual environment (created after setup)
└── README.md              # This file
```

## Dependencies
- pandas: Data manipulation
- yfinance: Yahoo Finance data
- numpy: Numerical computing
- quantstats: Portfolio analytics
- matplotlib: Plotting and visualizationPortfolio-AnalyseKI
In diesen Programm können sie Aktienportfolien anlegen und diese mit dem S&amp;P 500 vergleichen lassen.
