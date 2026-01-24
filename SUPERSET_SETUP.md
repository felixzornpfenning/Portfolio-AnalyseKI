# Apache Superset Dashboard Setup Guide

This guide will help you set up Apache Superset to create interactive dashboards for your portfolio geographic analysis.

## 📋 Prerequisites

1. **Python 3.8+** (already installed in your venv)
2. **SQLite Database** (created automatically by the script)
3. **Apache Superset** (will be installed)

## 🚀 Step 1: Install Apache Superset

Run these commands in your virtual environment:

```bash
# Activate your virtual environment
venv\Scripts\activate

# Install Superset
pip install apache-superset

# Install additional dependencies for better functionality
pip install sqlalchemy-utils
pip install pillow
```

## 🔧 Step 2: Initialize Superset

```bash
# Set environment variable for Superset configuration
$env:SUPERSET_CONFIG_PATH = "C:\Users\herbe\Portfolio-AnalyseKI\superset_config.py"

# Initialize Superset database
superset db upgrade

# Create admin user (you'll be prompted for username, email, password)
superset fab create-admin

# Load examples (optional)
superset load_examples

# Initialize Superset
superset init
```

## 🗃️ Step 3: Connect to Your Portfolio Database

1. **Run your Portfolio Analysis script** - this creates `portfolio_data.db`
2. **Start Superset**:
   ```bash
   superset run -p 8088 --with-threads --reload --debugger
   ```
3. **Open browser** to `http://localhost:8088`
4. **Login** with your admin credentials
5. **Add Database Connection**:
   - Go to "Settings" → "Database Connections"
   - Click "+" to add database
   - **SQLAlchemy URI**: `sqlite:///C:\Users\herbe\Portfolio-AnalyseKI\portfolio_data.db`
   - Test connection and save

## 📊 Step 4: Create Your Dashboard

### 🌍 World Map Chart
1. **Create Dataset**:
   - Go to "Data" → "Datasets" → "+"
   - Select your database and `country_allocation` table
   - Save as "Country Allocation"

2. **Create World Map**:
   - Go to "Charts" → "+"
   - Select "Country Allocation" dataset
   - Chart Type: "World Map"
   - Configuration:
     - **Entity**: `country`
     - **Metric**: `weight`
     - **Filters**: Add filter for `portfolio_name`

### 🥧 Pie Chart
1. **Create Pie Chart**:
   - Chart Type: "Pie Chart"
   - **Dimension**: `country`
   - **Metric**: `weight`
   - **Filters**: Add filter for `portfolio_name`

### 📊 Bar Chart
1. **Create Bar Chart**:
   - Chart Type: "Bar Chart"
   - **Dimension**: `country`
   - **Metric**: `weight`
   - **Filters**: Add filter for `portfolio_name`

## 🎛️ Step 5: Create Dashboard with Filters

1. **Create New Dashboard**:
   - Go to "Dashboards" → "+"
   - Name: "Portfolio Geographic Analysis"

2. **Add Charts** to dashboard

3. **Add Filter Box**:
   - Add a "Filter Box" chart
   - Configure to filter by `portfolio_name`
   - This creates your dropdown menu

## 🎨 Step 6: Customize Colors

Each country will automatically get different colors, but you can customize:

1. **In Chart Settings**:
   - Go to "Customize" tab
   - Set color scheme (e.g., "Superset Colors")
   - Ensure "Use consistent colors" is checked

## 🔄 Step 7: Auto-Refresh Data

To keep your dashboard updated:

1. **Set up automatic data refresh**:
   - In dataset settings, configure cache timeout
   - Run your Python script regularly to update data

2. **Refresh Dashboard**:
   - Use the refresh button in Superset
   - Or set auto-refresh intervals

## 🎯 Final Dashboard Features

Your dashboard will have:

✅ **Interactive World Map** - Click countries to drill down
✅ **Pie Chart** - Shows percentage allocation by country  
✅ **Bar Chart** - Easy comparison of country weights
✅ **Portfolio Dropdown** - Switch between different portfolios
✅ **Consistent Colors** - Same country = same color across all charts
✅ **Real-time Filters** - All charts update together

## 🛠️ Troubleshooting

### Common Issues:

1. **Database Connection Issues**:
   - Ensure the database path is correct and absolute
   - Check file permissions

2. **Chart Not Loading**:
   - Verify your dataset has data
   - Check column names match exactly

3. **Colors Not Consistent**:
   - Enable "Use consistent colors" in chart settings
   - Use the same color palette across charts

### 📁 Files Created:
- `portfolio_data.db` - SQLite database with your data
- `superset_dashboard_config.json` - Dashboard configuration
- `superset_config.py` - Superset configuration (if created)

## 🎉 Next Steps

1. **Run your enhanced portfolio script**
2. **Follow this setup guide**
3. **Create your dashboard**
4. **Enjoy interactive portfolio analysis!**

The dashboard will show you exactly where your investments are located geographically and how much weight each country has in your portfolio! 🌍📊