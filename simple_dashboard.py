"""
Simple Portfolio Dashboard using Flask
Alternative to Apache Superset with interactive visualizations
"""
import sqlite3
import pandas as pd
from flask import Flask, render_template, jsonify
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json

app = Flask(__name__)
DATABASE_PATH = 'portfolio_data.db'

def get_portfolio_data():
    """Get all portfolio data from the database"""
    conn = sqlite3.connect('portfolio_data.db')
    
    # Get portfolio holdings
    holdings_df = pd.read_sql_query("SELECT * FROM portfolio_holdings", conn)
    
    # Get country allocation
    country_df = pd.read_sql_query("SELECT * FROM country_allocation", conn)
    
    conn.close()
    return holdings_df, country_df

def get_country_color_mapping(country_df):
    """Get consistent color mapping for countries (same as pie chart)"""
    # Use Plotly's default color sequence to match pie chart
    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',
        '#dbdb8d', '#9edae5'
    ]
    
    # Prepare data same way as pie chart for consistency
    country_df_pct = country_df.copy()
    country_df_pct['percentage'] = country_df_pct['weight'] * 100
    
    # Apply same logic as pie chart for grouping
    threshold = 2.0
    
    # Get "Unknown" (always shown separately)
    unknown_data = country_df_pct[country_df_pct['country'] == 'Unknown']
    
    # Get "Other" (will be merged with small countries)
    other_data = country_df_pct[country_df_pct['country'] == 'Other']
    
    # Get regular countries (not "Other" or "Unknown")
    regular_countries = country_df_pct[~country_df_pct['country'].isin(['Other', 'Unknown'])]
    
    # Split regular countries into major (>= 2%) and minor (< 2%)
    major_countries = regular_countries[regular_countries['percentage'] >= threshold]
    minor_countries = regular_countries[regular_countries['percentage'] < threshold]
    
    # Consolidate: "Other" (from DB) + minor countries (< 2%) = consolidated "Other"
    if not minor_countries.empty or not other_data.empty:
        other_total = minor_countries['percentage'].sum()
        if not other_data.empty:
            other_total += other_data['percentage'].sum()
        
        consolidated_other = pd.DataFrame({
            'country': ['Other'],
            'percentage': [other_total]
        })
        chart_data = pd.concat([
            major_countries[['country', 'percentage']], 
            consolidated_other,
            unknown_data[['country', 'percentage']]
        ], ignore_index=True)
    else:
        chart_data = pd.concat([
            major_countries[['country', 'percentage']], 
            unknown_data[['country', 'percentage']]
        ], ignore_index=True)
    
    chart_data = chart_data.sort_values('percentage', ascending=False)
    
    # Create color mapping
    color_mapping = {}
    for i, country in enumerate(chart_data['country']):
        color_mapping[country] = plotly_colors[i % len(plotly_colors)]
    
    return color_mapping

def create_world_map(country_df):
    """Create an interactive world map showing portfolio allocation by country"""
    # Create a mapping from country names to ISO codes for the map
    country_codes = {
        'United States': 'USA', 'Germany': 'DEU', 'United Kingdom': 'GBR', 
        'France': 'FRA', 'Italy': 'ITA', 'Spain': 'ESP', 'Netherlands': 'NLD',
        'Switzerland': 'CHE', 'Japan': 'JPN', 'China': 'CHN', 'Canada': 'CAN',
        'Australia': 'AUS', 'South Korea': 'KOR', 'Taiwan': 'TWN', 'India': 'IND',
        'Brazil': 'BRA', 'Mexico': 'MEX', 'Sweden': 'SWE', 'Denmark': 'DNK',
        'Norway': 'NOR', 'Finland': 'FIN', 'Belgium': 'BEL', 'Austria': 'AUT',
        'Ireland': 'IRL', 'Israel': 'ISR', 'South Africa': 'ZAF', 'Saudi Arabia': 'SAU',
        'Hong Kong': 'HKG', 'Singapore': 'SGP', 'Thailand': 'THA', 'Malaysia': 'MYS',
        'Poland': 'POL', 'Greece': 'GRC'
    }
    
    # Get consistent color mapping
    color_mapping = get_country_color_mapping(country_df)
    
    # Add country codes to dataframe
    country_df_map = country_df.copy()
    country_df_map['country_code'] = country_df_map['country'].map(country_codes)
    country_df_map['percentage'] = country_df_map['weight'] * 100
    
    # Determine which countries should be colored as "Other" (threshold logic)
    threshold = 2.0
    
    def get_country_color(row):
        country = row['country']
        percentage = row['percentage']
        
        # If country is in the color mapping (major countries or Unknown), use that color
        if country in color_mapping:
            return color_mapping[country]
        # If country is below threshold and not Unknown/Other, use "Other" color
        elif percentage < threshold and country not in ['Unknown', 'Other']:
            return color_mapping.get('Other', '#ff7f0e')  # Orange for "Other"
        else:
            return '#1f77b4'  # Default blue
    
    # Add colors based on pie chart logic
    country_df_map['color'] = country_df_map.apply(get_country_color, axis=1)
    
    # Filter out countries without codes
    country_df_map = country_df_map.dropna(subset=['country_code'])
    
    # Create choropleth with custom colors to match pie chart
    fig = go.Figure()
    
    # Add each country separately to control colors
    for _, row in country_df_map.iterrows():
        if pd.notna(row['color']):  # Only add if color is valid
            fig.add_trace(go.Choropleth(
                locations=[row['country_code']],
                z=[row['percentage']],
                text=row['country'],
                colorscale=[[0, row['color']], [1, row['color']]],  # Single color for each country
                showscale=False,
                hovertemplate=f'<b>{row["country"]}</b><br>Allocation: {row["percentage"]:.2f}%<extra></extra>',
                marker_line_color='darkgray',
                marker_line_width=0.5,
            ))
    
    # Add a colorbar reference using the actual data range
    if not country_df_map.empty:
        max_val = country_df_map['percentage'].max()
        fig.add_trace(go.Choropleth(
            locations=[country_df_map.iloc[0]['country_code']],  # Use first valid location
            z=[max_val],  # Use max value for scale
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Portfolio %", x=1.02),
            visible=False  # Hide this trace but keep colorbar
        ))

    fig.update_layout(
        title={
            'text': 'Portfolio Geographic Allocation',
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=500
    )

    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_pie_chart(country_df):
    """Create pie chart of country allocation"""
    # Show all countries, don't group into "Others" since we already have "Other" and "Unknown" categories
    country_df_pct = country_df.copy()
    country_df_pct['percentage'] = country_df_pct['weight'] * 100
    
    # Debug: Log Denmark value being sent
    denmark_rows = country_df_pct[country_df_pct['country'].str.contains('Denmark', na=False)]
    if not denmark_rows.empty:
        denmark_pct = denmark_rows['percentage'].iloc[0]
        denmark_decimal = denmark_pct / 100
        print(f"DEBUG: Denmark percentage: {denmark_pct}% -> decimal sent to frontend: {denmark_decimal}")
    
    # For pie chart: Keep "Unknown" separate, consolidate small countries into "Other"
    threshold = 2.0
    
    # Get "Unknown" (always shown separately)
    unknown_data = country_df_pct[country_df_pct['country'] == 'Unknown']
    
    # Get "Other" (will be merged with small countries)
    other_data = country_df_pct[country_df_pct['country'] == 'Other']
    
    # Get regular countries (not "Other" or "Unknown")
    regular_countries = country_df_pct[~country_df_pct['country'].isin(['Other', 'Unknown'])]
    
    # Split regular countries into major (>= 2%) and minor (< 2%)
    major_countries = regular_countries[regular_countries['percentage'] >= threshold]
    minor_countries = regular_countries[regular_countries['percentage'] < threshold]
    
    # Consolidate: "Other" (from DB) + minor countries (< 2%) = consolidated "Other"
    if not minor_countries.empty or not other_data.empty:
        other_total = minor_countries['percentage'].sum()
        if not other_data.empty:
            other_total += other_data['percentage'].sum()
        
        consolidated_other = pd.DataFrame({
            'country': ['Other'],
            'percentage': [other_total]
        })
        # Combine: major countries + consolidated "Other" + "Unknown"
        chart_data = pd.concat([
            major_countries[['country', 'percentage']], 
            consolidated_other,
            unknown_data[['country', 'percentage']]
        ], ignore_index=True)
    else:
        # No "Other" or minor countries, just show major countries + "Unknown"
        chart_data = pd.concat([
            major_countries[['country', 'percentage']], 
            unknown_data[['country', 'percentage']]
        ], ignore_index=True)
    
    # Sort by percentage descending
    chart_data = chart_data.sort_values('percentage', ascending=False)
    
    # Use manual Plotly graph objects instead of express to avoid binary encoding issues
    # Use our exact percentage values and override Plotly's calculations completely
    percentage_values = chart_data['percentage'].tolist()
    labels = chart_data['country'].tolist()
    
    # Get consistent colors (same as world map)
    color_mapping = get_country_color_mapping(country_df)
    colors = [color_mapping.get(country, '#1f77b4') for country in labels]
    
    # Create custom text labels with our exact percentages
    custom_text = [f"{pct:.1f}%" for pct in percentage_values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=percentage_values,  # Use our exact percentages as values
        hole=0.3,
        textinfo='text',  # Show only our custom text (percentages), no labels
        text=custom_text,  # Force display of our exact percentages
        textposition='auto',
        marker=dict(colors=colors),  # Apply consistent colors
        hovertemplate='<b>%{label}</b><br>%{text}<br>Exact: %{value:.2f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Top Countries per Portfolio Allocation',
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
        height=700,  # Increased height to accommodate more legend items
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        font=dict(size=12)
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_holdings_pie_chart(portfolio_name):
    """Create pie chart of portfolio holdings"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Handle different portfolio selections
        if portfolio_name in ['All Portfolios', 'All Short Term', 'All Long Term']:
            if portfolio_name == 'All Portfolios':
                holdings_query = """
                SELECT symbol, name, SUM(weight) as weight 
                FROM portfolio_holdings 
                GROUP BY symbol, name
                ORDER BY weight DESC
                """
                holdings_df = pd.read_sql_query(holdings_query, conn)
            elif portfolio_name == 'All Short Term':
                holdings_query = """
                SELECT symbol, name, SUM(weight) as weight 
                FROM portfolio_holdings 
                WHERE portfolio_name LIKE '%Short%' 
                GROUP BY symbol, name
                ORDER BY weight DESC
                """
                holdings_df = pd.read_sql_query(holdings_query, conn)
            elif portfolio_name == 'All Long Term':
                holdings_query = """
                SELECT symbol, name, SUM(weight) as weight 
                FROM portfolio_holdings 
                WHERE portfolio_name LIKE '%Long%' 
                GROUP BY symbol, name
                ORDER BY weight DESC
                """
                holdings_df = pd.read_sql_query(holdings_query, conn)
            
            # Normalize aggregated weights to sum to 1.0 (100%)
            if not holdings_df.empty:
                total_weight = holdings_df['weight'].sum()
                if total_weight > 0:
                    holdings_df['weight'] = holdings_df['weight'] / total_weight
        else:
            # Individual portfolio
            holdings_query = """
            SELECT symbol, name, weight 
            FROM portfolio_holdings 
            WHERE portfolio_name = ? 
            ORDER BY weight DESC
            """
            holdings_df = pd.read_sql_query(holdings_query, conn, params=[portfolio_name])
        
        conn.close()
        
        if holdings_df.empty:
            return create_empty_chart('No holdings data available')
        
        # Convert to percentages
        holdings_df['percentage'] = holdings_df['weight'] * 100
        
        # Group smaller holdings (less than 2%) into "Other"
        threshold = 2.0
        major_holdings = holdings_df[holdings_df['percentage'] >= threshold]
        minor_holdings = holdings_df[holdings_df['percentage'] < threshold]
        
        # Prepare chart data
        if not minor_holdings.empty:
            other_total = minor_holdings['percentage'].sum()
            # Create labels using company names instead of ticker symbols
            labels = [row['name'] for _, row in major_holdings.iterrows()]
            values = major_holdings['percentage'].tolist()
            
            # Add "Other" category
            labels.append('Other')
            values.append(other_total)
        else:
            labels = [row['name'] for _, row in major_holdings.iterrows()]
            values = major_holdings['percentage'].tolist()
        
        # Create custom text labels with percentages
        custom_text = [f"{val:.1f}%" for val in values]
        
        # Custom color palette for holdings chart (different from country/sector charts)
        holdings_colors = [
            '#FF6B6B',  # Coral Red
            '#4ECDC4',  # Turquoise
            '#45B7D1',  # Sky Blue
            '#96CEB4',  # Mint Green
            '#FFEAA7',  # Warm Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Aquamarine
            '#F7DC6F',  # Light Yellow
            '#BB8FCE',  # Light Purple
            '#85C1E9',  # Light Blue
            '#F8C471',  # Light Orange
            '#82E0AA',  # Light Green
            '#F1948A',  # Light Red
            '#D7DBDD',  # Light Gray
            '#AED6F1'   # Pale Blue
        ]
        
        # Create pie chart moved further to the right with same size as others
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,  # Use percentage values directly
            hole=0.3,
            textinfo='text',  # Show only custom text
            text=custom_text,  # Custom percentage labels
            textposition='auto',
            marker=dict(colors=holdings_colors[:len(labels)]),  # Apply custom colors
            hovertemplate='<b>%{label}</b><br>%{text}<br>Exact: %{value:.2f}%<extra></extra>',
            domain={'x': [0.47, 0.98], 'y': [0.0, 1.0]}  # Fine-tuned positioning
        )])
        
        fig.update_layout(
            title={
                'text': 'Portfolio Holdings Allocation',
                'x': 0.5,  # Center the title
                'xanchor': 'center'
            },
            height=500,
            showlegend=True,
            font=dict(size=12)
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error creating holdings pie chart: {e}")
        return create_empty_chart('Error loading holdings data')

def get_etf_sector_allocations():
    """Define ETF sector allocations matching the analyzer"""
    return {
        # S&P 500 ETFs
        'SPY': {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
        'VUSA.L': {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
        'CSSPX.MI': {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
        
        # World/Global ETFs
        'VWRD.L': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
        'IWDA.L': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
        'VWCE.DE': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
        
        # Technology ETFs
        'SOXX': {'Technology': 1.0},
        'SEMI.AS': {'Technology': 1.0},
        'AIAI.MI': {'Technology': 0.85, 'Healthcare': 0.10, 'Communication Services': 0.05},
        'AIAI.SW': {'Technology': 0.85, 'Healthcare': 0.10, 'Communication Services': 0.05},
        'WISE': {'Technology': 1.0},
        'XDWT.L': {'Technology': 1.0},
        
        # Emerging Markets
        'EMIM.L': {'Technology': 0.20, 'Financial Services': 0.22, 'Consumer Cyclical': 0.15, 'Communication Services': 0.12, 'Energy': 0.08, 'Materials': 0.08, 'Healthcare': 0.06, 'Industrials': 0.05, 'Utilities': 0.04},
        'XMME.L': {'Technology': 0.20, 'Financial Services': 0.22, 'Consumer Cyclical': 0.15, 'Communication Services': 0.12, 'Energy': 0.08, 'Materials': 0.08, 'Healthcare': 0.06, 'Industrials': 0.05, 'Utilities': 0.04},
        
        # Bonds
        'IGLO.L': {'Government Bonds': 1.0},
        'IEAC.L': {'Corporate Bonds': 1.0},
        
        # Sector Specific
        'HEAL.L': {'Healthcare': 1.0},
        'INRG.L': {'Energy': 0.60, 'Utilities': 0.40},
        'EUAD': {'Industrials': 1.0},
        
        # Regional/Country ETFs
        'GREK': {'Financial Services': 0.35, 'Energy': 0.20, 'Materials': 0.15, 'Industrials': 0.12, 'Utilities': 0.10, 'Consumer Cyclical': 0.08},
        'SPOL.L': {'Financial Services': 0.30, 'Energy': 0.15, 'Industrials': 0.20, 'Technology': 0.15, 'Materials': 0.10, 'Consumer Cyclical': 0.10},
        'EXSA.DE': {'Technology': 0.18, 'Healthcare': 0.15, 'Financial Services': 0.14, 'Consumer Cyclical': 0.13, 'Industrials': 0.12, 'Consumer Defensive': 0.10, 'Energy': 0.08, 'Materials': 0.06, 'Utilities': 0.04},
        
        # Specialized ETFs
        'BNXG.DE': {'Technology': 0.70, 'Financial Services': 0.30},
        'EQAC.MI': {'Technology': 0.50, 'Consumer Cyclical': 0.15, 'Communication Services': 0.12, 'Healthcare': 0.10, 'Consumer Defensive': 0.08, 'Industrials': 0.05}
    }

def calculate_expanded_sector_allocation(holdings_df):
    """Calculate sector allocation including ETF decomposition"""
    etf_allocations = get_etf_sector_allocations()
    sector_weights = {}
    
    for _, row in holdings_df.iterrows():
        symbol = row['symbol']
        weight = row['weight']
        sector = row['sector']
        
        # Check if this is an ETF that we need to decompose
        if symbol in etf_allocations:
            # Decompose ETF into its sector allocations
            for etf_sector, etf_sector_weight in etf_allocations[symbol].items():
                adjusted_weight = weight * etf_sector_weight
                if etf_sector in sector_weights:
                    sector_weights[etf_sector] += adjusted_weight
                else:
                    sector_weights[etf_sector] = adjusted_weight
        else:
            # Regular stock - use its sector directly
            if sector in sector_weights:
                sector_weights[sector] += weight
            else:
                sector_weights[sector] = weight
    
    return sector_weights

def create_sector_chart(holdings_df):
    """Create sector allocation chart with ETF decomposition"""
    try:
        # Calculate expanded sector allocation
        sector_weights = calculate_expanded_sector_allocation(holdings_df)
        
        # Convert to percentages and create DataFrame for plotting
        sector_data = []
        for sector, weight in sector_weights.items():
            percentage = weight * 100
            sector_data.append({'sector': sector, 'percentage': percentage})
        
        # Sort by percentage
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values('percentage', ascending=False)
        
        # Custom darker color palette for sector chart with transparency
        sector_colors = [
            'rgba(44, 62, 80, 0.8)',    # Dark Blue Gray with 80% opacity
            'rgba(139, 69, 19, 0.8)',   # Saddle Brown with 80% opacity
            'rgba(72, 61, 139, 0.8)',   # Dark Slate Blue with 80% opacity
            'rgba(47, 79, 79, 0.8)',    # Dark Slate Gray with 80% opacity
            'rgba(128, 0, 128, 0.8)',   # Purple with 80% opacity
            'rgba(178, 34, 34, 0.8)',   # Fire Brick with 80% opacity
            'rgba(70, 130, 180, 0.8)',  # Steel Blue with 80% opacity
            'rgba(34, 139, 34, 0.8)',   # Forest Green with 80% opacity
            'rgba(205, 133, 63, 0.8)',  # Peru with 80% opacity
            'rgba(75, 0, 130, 0.8)',    # Indigo with 80% opacity
            'rgba(139, 0, 139, 0.8)',   # Dark Magenta with 80% opacity
            'rgba(85, 107, 47, 0.8)',   # Dark Olive Green with 80% opacity
            'rgba(160, 82, 45, 0.8)',   # Sienna with 80% opacity
            'rgba(25, 25, 112, 0.8)',   # Midnight Blue with 80% opacity
            'rgba(139, 0, 0, 0.8)',     # Dark Red with 80% opacity
            'rgba(0, 100, 0, 0.8)'      # Dark Green with 80% opacity
        ]
        
        # Create pie chart using Plotly Graph Objects for consistency
        fig = go.Figure(data=[go.Pie(
            labels=sector_df['sector'],
            values=sector_df['percentage'],
            hole=0.3,
            textinfo='percent',  # Only show percentages on slices
            textposition='auto',
            marker=dict(colors=sector_colors[:len(sector_df)]),  # Apply custom dark colors
            hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Portfolio Sector Allocation (Including ETF Breakdown)',
                'x': 0.5,  # Center the title
                'xanchor': 'center'
            },
            height=500,
            showlegend=True,  # Keep legend for sector names
            font=dict(size=12)
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error creating sector chart: {e}")
        return create_empty_chart('Error loading sector data')

def create_empty_chart(message):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    holdings_df, country_df = get_portfolio_data()
    
    # Get unique portfolios for dropdown and create grouped options
    individual_portfolios = holdings_df['portfolio_name'].unique().tolist()
    
    # Create dropdown options with groups
    portfolios = ['All Portfolios', 'All Short Term', 'All Long Term'] + individual_portfolios
    
    # Create charts for all portfolios initially - need to normalize country data
    # Aggregate country allocation for all portfolios
    portfolio_countries = country_df.groupby('country')['weight'].sum().reset_index()
    # Normalize weights to sum to 1.0 (100%)
    total_weight = portfolio_countries['weight'].sum()
    if total_weight > 0:
        portfolio_countries['weight'] = portfolio_countries['weight'] / total_weight
    
    world_map = create_world_map(portfolio_countries)
    pie_chart = create_pie_chart(portfolio_countries)
    holdings_pie_chart = create_holdings_pie_chart('All Portfolios')  # Default to all portfolios
    sector_chart = create_sector_chart(holdings_df)
    
    return render_template('dashboard.html',
                         portfolios=portfolios,
                         world_map=world_map,
                         pie_chart=pie_chart,
                         holdings_pie_chart=holdings_pie_chart,
                         sector_chart=sector_chart)

@app.route('/api/portfolio/<portfolio_name>')
def get_portfolio_charts(portfolio_name):
    """API endpoint to get charts for a specific portfolio or portfolio group"""
    holdings_df, country_df = get_portfolio_data()
    
    if portfolio_name == 'All Portfolios':
        # Use all portfolios
        portfolio_holdings = holdings_df
        # Aggregate country allocation for all portfolios
        portfolio_countries = country_df.groupby('country')['weight'].sum().reset_index()
        # Normalize weights to sum to 1.0 (100%)
        total_weight = portfolio_countries['weight'].sum()
        if total_weight > 0:
            portfolio_countries['weight'] = portfolio_countries['weight'] / total_weight
        portfolio_countries['portfolio_name'] = 'All Portfolios'
        portfolio_countries['date_analyzed'] = country_df['date_analyzed'].iloc[0]
        
    elif portfolio_name == 'All Short Term':
        # Filter for short-term portfolios (portfolios ending with "Short Term")
        short_term_portfolios = [p for p in holdings_df['portfolio_name'].unique() if p.endswith('Short Term')]
        
        if short_term_portfolios:
            portfolio_holdings = holdings_df[holdings_df['portfolio_name'].isin(short_term_portfolios)]
            # Aggregate country allocation for short-term portfolios
            portfolio_countries = country_df[country_df['portfolio_name'].isin(short_term_portfolios)]
            portfolio_countries = portfolio_countries.groupby('country')['weight'].sum().reset_index()
            # Normalize weights to sum to 1.0 (100%)
            total_weight = portfolio_countries['weight'].sum()
            if total_weight > 0:
                portfolio_countries['weight'] = portfolio_countries['weight'] / total_weight
            portfolio_countries['portfolio_name'] = 'All Short Term'
            portfolio_countries['date_analyzed'] = country_df['date_analyzed'].iloc[0]
        else:
            portfolio_holdings = pd.DataFrame()
            portfolio_countries = pd.DataFrame()
            
    elif portfolio_name == 'All Long Term':
        # Filter for long-term portfolios (portfolios ending with "Long Term")
        long_term_portfolios = [p for p in holdings_df['portfolio_name'].unique() if p.endswith('Long Term')]
        
        if long_term_portfolios:
            portfolio_holdings = holdings_df[holdings_df['portfolio_name'].isin(long_term_portfolios)]
            # Aggregate country allocation for long-term portfolios
            portfolio_countries = country_df[country_df['portfolio_name'].isin(long_term_portfolios)]
            portfolio_countries = portfolio_countries.groupby('country')['weight'].sum().reset_index()
            # Normalize weights to sum to 1.0 (100%)
            total_weight = portfolio_countries['weight'].sum()
            if total_weight > 0:
                portfolio_countries['weight'] = portfolio_countries['weight'] / total_weight
            portfolio_countries['portfolio_name'] = 'All Long Term'
            portfolio_countries['date_analyzed'] = country_df['date_analyzed'].iloc[0]
        else:
            portfolio_holdings = pd.DataFrame()
            portfolio_countries = pd.DataFrame()
            
    else:
        # Filter for specific individual portfolio
        portfolio_holdings = holdings_df[holdings_df['portfolio_name'] == portfolio_name]
        portfolio_countries = country_df[country_df['portfolio_name'] == portfolio_name]
    
    # Create charts with filtered data
    if not portfolio_countries.empty and not portfolio_holdings.empty:
        charts = {
            'world_map': create_world_map(portfolio_countries),
            'pie_chart': create_pie_chart(portfolio_countries),
            'holdings_pie_chart': create_holdings_pie_chart(portfolio_name),
            'sector_chart': create_sector_chart(portfolio_holdings)
        }
    else:
        # Return empty charts if no data
        charts = {
            'world_map': create_empty_chart('No geographic data available'),
            'pie_chart': create_empty_chart('No country data available'),
            'holdings_pie_chart': create_empty_chart('No holdings data available'),
            'sector_chart': create_empty_chart('No sector data available')
        }
    
    return jsonify(charts)

@app.route('/api/summary')
def get_summary():
    """Get portfolio summary statistics"""
    holdings_df, country_df = get_portfolio_data()
    
    # Count short-term and long-term portfolios
    portfolios = holdings_df['portfolio_name'].unique()
    short_term_count = sum(1 for p in portfolios if p.endswith('Short Term'))
    long_term_count = sum(1 for p in portfolios if p.endswith('Long Term'))
    
    summary = {
        'total_portfolios': len(portfolios),
        'short_term_portfolios': short_term_count,
        'long_term_portfolios': long_term_count,
        'total_holdings': len(holdings_df),
        'total_countries': len(country_df),
        'top_country': country_df.nlargest(1, 'weight')['country'].iloc[0] if not country_df.empty else 'N/A',
        'top_country_pct': round(country_df.nlargest(1, 'weight')['weight'].iloc[0] * 100, 2) if not country_df.empty else 0
    }
    
    return jsonify(summary)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)