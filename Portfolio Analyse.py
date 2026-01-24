# Portfolio Analyse mit QuantStats
import pandas as pd
import yfinance as yf
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

# GICS DATA SOURCES:
# 1. Yahoo Finance (✅ Implemented) - Free, good coverage for stocks
# 2. Alpha Vantage API - Free tier: 25 requests/day
# 3. Finnhub API - Free tier: 60 calls/minute  
# 4. Polygon.io - Paid service, comprehensive data
# 5. Bloomberg API - Professional, paid service
# 6. Refinitiv (Thomson Reuters) - Professional, paid service

# ETF SECTOR DATA SOURCES (More Accurate):
# 1. ETF Provider APIs (iShares, Vanguard, SPDR) - Most accurate
# 2. Morningstar API - Comprehensive ETF analytics
# 3. ETF Database (etfdb.com) - Web scraping possible
# 4. State Street Global Advisors API - For SPDR ETFs
# 5. BlackRock API - For iShares ETFs

# Optional imports for additional providers:
# import requests  # For ETF provider APIs or web scraping
# import finnhub   # pip install finnhub-python
# from bs4 import BeautifulSoup  # pip install beautifulsoup4 (for web scraping)

# For Superset dashboard integration:
import sqlite3
import json
warnings.filterwarnings('ignore')
qs.extend_pandas()

class PortfolioAnalyzer:
    def __init__(self, portfolio_stocks, portfolio_weights=None, start_date=None, end_date=None):
        """Initialize the portfolio analyzer using QuantStats."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.start_date = start_date
        self.end_date = end_date
        if isinstance(portfolio_stocks, dict):
            self.stocks = list(portfolio_stocks.keys())
            self.weights = list(portfolio_stocks.values())
        else:
            self.stocks = portfolio_stocks
            if portfolio_weights:
                self.weights = portfolio_weights
            else:
                self.weights = [1/len(self.stocks)] * len(self.stocks)
        self.weights = np.array(self.weights) / sum(self.weights)
        print(f"Portfolio: {dict(zip(self.stocks, self.weights))}")
        print(f"Analysis period: {start_date} to {end_date}")

    def fetch_data(self):
        """Fetch stock data and S&P 500 data."""
        print("Fetching data...")
        portfolio_data = {}
        for stock in self.stocks:
            try:
                ticker = yf.Ticker(stock)
                data = ticker.history(start=self.start_date, end=self.end_date)
                info = ticker.info
                name = info.get('longName') or info.get('shortName') or stock
                
                # Extract geographic information
                country = self._get_stock_country(stock, info)
                
                # Detect if it's an ETF
                fund_family = info.get('fundFamily', '')
                quote_type = info.get('quoteType', '')
                is_etf = quote_type == 'ETF' or 'ETF' in name.upper() or any(x in name.upper() for x in ['FUND', 'INDEX'])
                
                if is_etf:
                    sector = 'ETF - Mixed'
                    industry = f"ETF - {info.get('category', 'Diversified')}"
                    etf_info = self._get_etf_sector_info(ticker, stock)
                else:
                    sector = info.get('sector', 'N/A')
                    industry = info.get('industry', 'N/A')
                    etf_info = None
                
                portfolio_data[stock] = data['Close']
                
                # Store sector/industry information
                if not hasattr(self, 'stock_info'):
                    self.stock_info = {}
                self.stock_info[stock] = {
                    'name': name,
                    'sector': sector,
                    'industry': industry,
                    'is_etf': is_etf,
                    'etf_sectors': etf_info,
                    'country': country
                }
                
                if is_etf:
                    print(f"✓ {name} ({stock}) - ETF - {info.get('category', 'Mixed')} - data fetched")
                else:
                    print(f"✓ {name} ({stock}) - Sector: {sector} - data fetched")
            except Exception as e:
                print(f"✗ Error fetching {stock}: {e}")
        try:
            sp500 = yf.Ticker("SPY")
            sp500_data = sp500.history(start=self.start_date, end=self.end_date)
            self.benchmark_data = sp500_data['Close']
            # Fill missing benchmark data too
            self.benchmark_data = self.benchmark_data.ffill().bfill()
            print("✓ S&P 500 (SPY) data fetched")
        except Exception as e:
            print(f"✗ Error fetching S&P 500: {e}")
        # Remove timezone info from each series before combining
        for stock in portfolio_data:
            if portfolio_data[stock].index.tz is not None:
                portfolio_data[stock].index = portfolio_data[stock].index.tz_localize(None)
        
        # Create DataFrame and handle missing data
        self.stock_data = pd.DataFrame(portfolio_data)
        
        # Forward-fill missing data (use last available price for missing dates)
        print(f"Missing data points before filling: {self.stock_data.isnull().sum().sum()}")
        self.stock_data = self.stock_data.ffill()  # Forward fill
        self.stock_data = self.stock_data.bfill()  # Backward fill for any remaining NaNs at the beginning
        
        # Only drop rows if ALL stocks are missing data for that date
        initial_shape = self.stock_data.shape
        self.stock_data = self.stock_data.dropna(how='all')
        print(f"Data shape: {initial_shape} -> {self.stock_data.shape}")
        
        if self.stock_data.empty:
            print("\n[WARNING] No overlapping data for all tickers in the selected date range. Try a more recent start_date or check ticker data on Yahoo Finance.")
        else:
            remaining_missing = self.stock_data.isnull().sum().sum()
            if remaining_missing > 0:
                print(f"[INFO] {remaining_missing} missing data points remain after filling")
        
        # Remove timezone information if present using pandas conversion
        if isinstance(self.stock_data.index, pd.DatetimeIndex):
            if self.stock_data.index.tz is not None:
                self.stock_data.index = self.stock_data.index.tz_localize(None)
        
        if isinstance(self.benchmark_data.index, pd.DatetimeIndex):
            if self.benchmark_data.index.tz is not None:
                self.benchmark_data.index = self.benchmark_data.index.tz_localize(None)
        
        return self.stock_data, self.benchmark_data

    def _get_stock_country(self, symbol, info):
        """Extract country information prioritizing company domicile over exchange location."""
        try:
            # Manual override for well-known stocks where Yahoo Finance data is incomplete
            known_stocks = {
                'MOH.MU': 'France',      # LVMH - French luxury conglomerate
                'NESM.SG': 'Switzerland', # Nestlé - Swiss multinational
                'ROG.SW': 'Switzerland',  # Roche - Swiss pharmaceutical
                'NOVO-B.CO': 'Denmark',   # Novo Nordisk - Danish pharmaceutical
                'NVO': 'Denmark',         # Novo Nordisk ADR
                'OR.PA': 'France',        # L'Oréal
                'MC.PA': 'France',        # LVMH Paris listing
                'ASML': 'Netherlands',    # ASML - Dutch semiconductor
            }
            
            if symbol in known_stocks:
                return known_stocks[symbol]
            
            # First priority: Get actual company country from Yahoo Finance info
            # This represents where the company is headquartered/incorporated
            company_country = info.get('country', '')
            if company_country and company_country != 'Unknown' and company_country != 'N/A':
                return company_country
            
            # Second priority: Check if it's an ETF with domicile info
            fund_family = info.get('fundFamily', '')
            domicile = info.get('domicile', '')
            
            if domicile:
                return domicile
            
            # For ETFs, try to get issuer country information
            if info.get('quoteType') == 'ETF':
                # Common ETF issuer country patterns
                etf_issuer_mapping = {
                    'iShares': self._get_ishares_domicile(symbol),
                    'Vanguard': self._get_vanguard_domicile(symbol),
                    'SPDR': 'United States',
                    'Invesco': self._get_invesco_domicile(symbol),
                    'Xtrackers': 'Germany',  # Deutsche Bank subsidiary
                    'Amundi': 'France',
                    'Lyxor': 'France',
                    'HSBC': self._get_hsbc_domicile(symbol),
                }
                
                name = info.get('longName', '').upper()
                for issuer, country in etf_issuer_mapping.items():
                    if issuer.upper() in name:
                        if callable(country):
                            result = country
                            if result != 'Unknown':
                                return result
                        else:
                            return country
            
            # Fallback: Exchange location mapping (only when company country unavailable)
            exchange_mapping = {
                # European exchanges
                '.L': 'United Kingdom', '.MI': 'Italy', '.PA': 'France',
                '.DE': 'Germany', '.MU': 'Germany', '.AS': 'Netherlands', '.SW': 'Switzerland',
                '.CO': 'Denmark', '.HE': 'Finland', '.ST': 'Sweden',
                '.OL': 'Norway', '.VI': 'Austria', '.BC': 'Spain',
                '.LS': 'Portugal', '.BR': 'Belgium', '.IR': 'Ireland',
                '.SG': 'Singapore',  # Singapore Stock Exchange
                
                # Other exchanges
                '.TO': 'Canada', '.TSE': 'Japan', '.HK': 'Hong Kong',
                '.SS': 'China', '.SZ': 'China', '.SI': 'Singapore',
                '.AX': 'Australia', '.NZ': 'New Zealand', '.SA': 'Brazil', '.MX': 'Mexico',
            }
            
            # Exchange code mapping (3-letter codes from Yahoo Finance)
            exchange_code_mapping = {
                'LSE': 'United Kingdom', 'LON': 'United Kingdom',  # London
                'MIL': 'Italy', 'BIT': 'Italy',  # Milan
                'PAR': 'France', 'EPA': 'France',  # Paris
                'FRA': 'Germany', 'GER': 'Germany', 'XETRA': 'Germany',  # Frankfurt/Xetra
                'MUN': 'Germany', 'STU': 'Germany',  # Munich/Stuttgart
                'AMS': 'Netherlands',  # Amsterdam
                'SWX': 'Switzerland', 'VTX': 'Switzerland',  # Switzerland
                'CPH': 'Denmark', 'CSE': 'Denmark',  # Copenhagen
                'HEL': 'Finland',  # Helsinki
                'STO': 'Sweden', 'OMX': 'Sweden',  # Stockholm
                'OSL': 'Norway',  # Oslo
                'VIE': 'Austria',  # Vienna
                'BME': 'Spain', 'MCE': 'Spain',  # Madrid
                'TSX': 'Canada', 'TOR': 'Canada',  # Toronto
                'JPX': 'Japan', 'TYO': 'Japan',  # Tokyo
                'HKG': 'Hong Kong', 'HKEX': 'Hong Kong',  # Hong Kong
                'SHA': 'China', 'SHE': 'China', 'SHZ': 'China',  # Shanghai/Shenzhen
                'SGX': 'Singapore', 'SES': 'Singapore',  # Singapore
                'ASX': 'Australia',  # Australia
                'NZX': 'New Zealand',  # New Zealand
                'SAO': 'Brazil', 'BMV': 'Brazil',  # São Paulo
                'MEX': 'Mexico', 'BMV': 'Mexico',  # Mexico
            }
            
            # Check exchange suffix for ETFs or unknown companies
            for suffix, country in exchange_mapping.items():
                if symbol.endswith(suffix):
                    return country
            
            # Check exchange code from Yahoo Finance
            exchange = info.get('exchange', '')
            if exchange:
                exchange_upper = exchange.upper()
                # Check US exchanges first
                if any(ex in exchange_upper for ex in ['NYSE', 'NASDAQ', 'BATS', 'ARCA', 'NYQ', 'NMS']):
                    return 'United States'
                # Check other exchange codes
                for code, country in exchange_code_mapping.items():
                    if code in exchange_upper:
                        return country
            
            # Default: assume US for simple symbols without suffixes
            if len(symbol) <= 5 and '.' not in symbol:
                return 'United States'
            
            return 'Unknown'
            
        except Exception as e:
            print(f"Error getting country for {symbol}: {e}")
            return 'Unknown'
    
    def _get_ishares_domicile(self, symbol):
        """Get iShares ETF domicile based on symbol patterns."""
        if '.L' in symbol or '.MI' in symbol or '.PA' in symbol or '.DE' in symbol:
            return 'Ireland'  # Most European iShares ETFs domiciled in Ireland
        return 'United States'  # US-listed iShares
    
    def _get_vanguard_domicile(self, symbol):
        """Get Vanguard ETF domicile based on symbol patterns."""
        if '.L' in symbol:
            return 'Ireland'  # Vanguard Europe ETFs
        elif '.TO' in symbol:
            return 'Canada'   # Vanguard Canada ETFs
        return 'United States'  # US Vanguard ETFs
    
    def _get_invesco_domicile(self, symbol):
        """Get Invesco ETF domicile based on symbol patterns."""
        if any(suffix in symbol for suffix in ['.L', '.MI', '.PA', '.DE']):
            return 'Ireland'  # European Invesco ETFs
        return 'United States'  # US Invesco ETFs
    
    def _get_hsbc_domicile(self, symbol):
        """Get HSBC ETF domicile based on symbol patterns."""
        if '.L' in symbol:
            return 'United Kingdom'  # HSBC UK ETFs
        return 'United Kingdom'  # Default HSBC domicile

    def _get_etf_sector_info(self, ticker, symbol):
        """Get ETF sector allocation information using comprehensive symbol and name matching."""
        try:
            info = ticker.info
            name = info.get('longName', '').upper()
            category = info.get('category', 'Unknown')
            
            # Comprehensive ETF database with real allocations
            etf_allocations = {
                # S&P 500 ETFs
                'CSSPX.MI': {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
                'SPY': {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
                'VUSA.L': {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
                
                # World/Global ETFs
                'VWRD.L': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
                'IWDA.L': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
                'VWCE.DE': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
                
                # Technology ETFs
                'SOXX': {'Technology': 1.0},
                'SEMI.AS': {'Technology': 1.0},
                'AIAI.MI': {'Technology': 0.85, 'Healthcare': 0.10, 'Communication Services': 0.05},
                'AIAI.SW': {'Technology': 0.85, 'Healthcare': 0.10, 'Communication Services': 0.05},
                'WISE': {'Technology': 1.0},  # AI ETF - classified as pure technology
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
                'EUAD': {'Industrials': 1.0},  # Aerospace & Defense
                
                # Regional/Country ETFs
                'GREK': {'Financial Services': 0.35, 'Energy': 0.20, 'Materials': 0.15, 'Industrials': 0.12, 'Utilities': 0.10, 'Consumer Cyclical': 0.08},
                'SPOL.L': {'Financial Services': 0.30, 'Energy': 0.15, 'Industrials': 0.20, 'Technology': 0.15, 'Materials': 0.10, 'Consumer Cyclical': 0.10},
                'EXSA.DE': {'Technology': 0.18, 'Healthcare': 0.15, 'Financial Services': 0.14, 'Consumer Cyclical': 0.13, 'Industrials': 0.12, 'Consumer Defensive': 0.10, 'Energy': 0.08, 'Materials': 0.06, 'Utilities': 0.04},
                
                # Specialized ETFs
                'BNXG.DE': {'Technology': 0.70, 'Financial Services': 0.30},  # Blockchain ETF
                'EQAC.MI': {'Technology': 0.50, 'Consumer Cyclical': 0.15, 'Communication Services': 0.12, 'Healthcare': 0.10, 'Consumer Defensive': 0.08, 'Industrials': 0.05}  # NASDAQ-100
            }
            
            # Direct symbol lookup
            if symbol in etf_allocations:
                return etf_allocations[symbol]
            
            # Pattern matching for unlisted ETFs
            symbol_upper = symbol.upper()
            name_patterns = {
                # S&P 500 patterns
                ('S&P', '500', 'SP500'): {'Technology': 0.31, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.09, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
                
                # World/MSCI World patterns
                ('WORLD', 'GLOBAL', 'MSCI WORLD'): {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
                
                # Technology patterns
                ('TECH', 'ARTIFICIAL INTELLIGENCE', 'AI', 'SEMICONDUCTOR', 'SOFTWARE'): {'Technology': 0.90, 'Communication Services': 0.10},
                
                # Healthcare patterns
                ('HEALTHCARE', 'BIOTECH', 'PHARMA', 'MEDICAL'): {'Healthcare': 1.0},
                
                # Financial patterns
                ('BANK', 'FINANCIAL', 'FINANCE'): {'Financial Services': 1.0},
                
                # Energy patterns
                ('ENERGY', 'OIL', 'CLEAN ENERGY', 'RENEWABLE'): {'Energy': 0.70, 'Utilities': 0.30},
                
                # Emerging Markets patterns
                ('EMERGING', 'EM ', 'MSCI EM'): {'Technology': 0.20, 'Financial Services': 0.22, 'Consumer Cyclical': 0.15, 'Communication Services': 0.12, 'Energy': 0.08, 'Materials': 0.08, 'Healthcare': 0.06, 'Industrials': 0.05, 'Utilities': 0.04},
                
                # Bond patterns
                ('BOND', 'GOVERNMENT', 'CORPORATE', 'TREASURY'): {'Government Bonds': 0.70, 'Corporate Bonds': 0.30},
                
                # European patterns
                ('EUROPE', 'EURO', 'STOXX'): {'Technology': 0.18, 'Healthcare': 0.15, 'Financial Services': 0.14, 'Consumer Cyclical': 0.13, 'Industrials': 0.12, 'Consumer Defensive': 0.10, 'Energy': 0.08, 'Materials': 0.06, 'Utilities': 0.04}
            }
            
            # Check patterns in both symbol and name
            search_text = f"{symbol_upper} {name}"
            for patterns, allocation in name_patterns.items():
                if any(pattern in search_text for pattern in patterns):
                    return allocation
            
            # If still unknown, try category-based matching
            category_mapping = {
                'Large Cap': {'Technology': 0.28, 'Healthcare': 0.13, 'Financial Services': 0.13, 'Communication Services': 0.11, 'Consumer Cyclical': 0.10, 'Industrials': 0.08, 'Consumer Defensive': 0.06, 'Energy': 0.04, 'Real Estate': 0.03, 'Materials': 0.03},
                'Foreign Large Cap': {'Technology': 0.24, 'Financial Services': 0.15, 'Healthcare': 0.12, 'Consumer Cyclical': 0.11, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.07, 'Energy': 0.05, 'Materials': 0.04, 'Real Estate': 0.04},
                'Technology': {'Technology': 1.0},
                'Healthcare': {'Healthcare': 1.0},
                'Financial': {'Financial Services': 1.0}
            }
            
            for cat_pattern, allocation in category_mapping.items():
                if cat_pattern.lower() in category.lower():
                    return allocation
            
            print(f"   ⚠️  Unknown ETF allocation for {symbol} ({name[:30]}...) - using diversified estimate")
            # Default diversified allocation for unknown ETFs
            return {'Technology': 0.25, 'Healthcare': 0.15, 'Financial Services': 0.15, 'Consumer Cyclical': 0.12, 'Industrials': 0.10, 'Communication Services': 0.08, 'Consumer Defensive': 0.08, 'Energy': 0.04, 'Materials': 0.03}
            
        except Exception as e:
            print(f"   ❌ Error determining ETF sector allocation for {symbol}: {e}")
            return {'Mixed/Unknown': 1.0}

    def calculate_portfolio_returns(self):
        """Calculate portfolio returns using weights."""
        stock_returns = self.stock_data.pct_change().dropna()
        self.portfolio_returns = (stock_returns * self.weights).sum(axis=1)
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()
        common_dates = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        self.portfolio_returns = self.portfolio_returns.loc[common_dates]
        self.benchmark_returns = self.benchmark_returns.loc[common_dates]
        return self.portfolio_returns, self.benchmark_returns

    def analyze_sector_allocation(self, portfolio_name=None):
        """Analyze portfolio sector allocation based on GICS sectors, including ETF breakdowns."""
        if not hasattr(self, 'stock_info'):
            print("No sector information available. Run fetch_data() first.")
            return
        
        print("\n" + "="*60)
        if portfolio_name:
            print(f"PORTFOLIO SECTOR ANALYSIS: {portfolio_name}")
        else:
            print("PORTFOLIO SECTOR ANALYSIS (GICS + ETF BREAKDOWN)")
        print("="*60)
        
        # Calculate sector weights (including ETF decomposition)
        sector_weights = {}
        etf_holdings = []
        
        for i, stock in enumerate(self.stocks):
            stock_info = self.stock_info[stock]
            weight = self.weights[i]
            
            if stock_info['is_etf'] and stock_info['etf_sectors']:
                # Decompose ETF into its sector allocations
                etf_holdings.append((stock, stock_info, weight))
                for sector, etf_sector_weight in stock_info['etf_sectors'].items():
                    adjusted_weight = weight * etf_sector_weight
                    if sector in sector_weights:
                        sector_weights[sector] += adjusted_weight
                    else:
                        sector_weights[sector] = adjusted_weight
            else:
                # Regular stock
                sector = stock_info['sector']
                if sector in sector_weights:
                    sector_weights[sector] += weight
                else:
                    sector_weights[sector] = weight
        
        # Display sector allocation
        print("\n📊 TOTAL SECTOR ALLOCATION (Including ETF Breakdown):")
        print("-" * 50)
        sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        
        for sector, weight in sorted_sectors:
            percentage = weight * 100
            print(f"{sector:.<35} {percentage:>6.2f}%")
        
        # Display ETF breakdowns
        if etf_holdings:
            print(f"\n🏦 ETF HOLDINGS & ESTIMATED SECTOR BREAKDOWN:")
            print("-" * 50)
            for stock, info, weight in etf_holdings:
                print(f"\n📈 {stock}: {info['name'][:40]}... ({weight*100:.1f}%)")
                print(f"    Category: {info['industry']}")
                if info['etf_sectors']:
                    print("    Estimated Sector Breakdown:")
                    for sector, etf_weight in info['etf_sectors'].items():
                        portfolio_contribution = weight * etf_weight * 100
                        etf_percentage = etf_weight * 100
                        print(f"      - {sector}: {etf_percentage:.1f}% of ETF = {portfolio_contribution:.2f}% of portfolio")
        
        # Display individual stock details by sector
        print(f"\n📋 INDIVIDUAL STOCKS BY SECTOR:")
        print("-" * 50)
        for sector, _ in sorted_sectors:
            if sector not in ['Mixed/Unknown', 'ETF - Mixed']:
                individual_stocks = [(stock, self.stock_info[stock], self.weights[i]) 
                               for i, stock in enumerate(self.stocks) 
                               if not self.stock_info[stock]['is_etf'] and self.stock_info[stock]['sector'] == sector]
                
                if individual_stocks:
                    print(f"\n🏢 {sector}:")
                    for stock, info, weight in individual_stocks:
                        print(f"  • {stock}: {info['name'][:45]}... ({weight*100:.1f}%)")
                        print(f"    Industry: {info['industry']}")

    def analyze_geographic_allocation(self, portfolio_name=None):
        """Analyze portfolio geographic allocation by country."""
        if not hasattr(self, 'stock_info'):
            print("No geographic information available. Run fetch_data() first.")
            return {}
        
        print("\n" + "="*60)
        if portfolio_name:
            print(f"PORTFOLIO GEOGRAPHIC ANALYSIS: {portfolio_name}")
        else:
            print("PORTFOLIO GEOGRAPHIC ANALYSIS")
        print("="*60)
        
        # Calculate country weights
        country_weights = {}
        
        for i, stock in enumerate(self.stocks):
            stock_info = self.stock_info[stock]
            weight = self.weights[i]
            
            if stock_info['is_etf']:
                # For ETFs, get detailed geographic exposure
                geographic_exposure = self._get_etf_geographic_exposure(stock, stock_info)
                if isinstance(geographic_exposure, dict):
                    # Distribute ETF weight across countries based on exposure
                    for country, exposure_pct in geographic_exposure.items():
                        country_weight = weight * (exposure_pct / 100.0)
                        if country in country_weights:
                            country_weights[country] += country_weight
                        else:
                            country_weights[country] = country_weight
                else:
                    # Fallback: treat as single country
                    country = geographic_exposure if geographic_exposure else stock_info['country']
                    if country in country_weights:
                        country_weights[country] += weight
                    else:
                        country_weights[country] = weight
            else:
                # For individual stocks, use company country
                country = stock_info['country']
                if country in country_weights:
                    country_weights[country] += weight
                else:
                    country_weights[country] = weight
        
        # Display geographic allocation
        print("\n🌍 GEOGRAPHIC ALLOCATION:")
        print("-" * 40)
        sorted_countries = sorted(country_weights.items(), key=lambda x: x[1], reverse=True)
        
        for country, weight in sorted_countries:
            percentage = weight * 100
            print(f"{country:.<25} {percentage:>6.2f}%")
        
        return dict(sorted_countries)

    def _get_etf_geographic_exposure(self, symbol, stock_info):
        """Get more accurate geographic exposure for ETFs based on their actual holdings."""
        name = stock_info['name'].upper()
        
        # Comprehensive ETF geographic mapping with primary exposure
        etf_geographic_mapping = {
            # US-focused ETFs
            'CSSPX.MI': {'United States': 100.0},  # S&P 500
            'SOXX': {'United States': 85.0, 'Taiwan': 10.0, 'Netherlands': 5.0},  # Semiconductor
            'SPY': {'United States': 100.0},
            'QQQ': {'United States': 100.0},
            'VUSA.L': {'United States': 100.0},  # S&P 500
            
            # Global/World ETFs (Updated 2024/2025 allocations)
            'VWRD.L': {'United States': 63.0, 'Japan': 5.8, 'United Kingdom': 3.8, 'Taiwan': 3.2, 'France': 3.0, 'Canada': 3.0, 'Switzerland': 2.8, 'Germany': 2.2, 'Netherlands': 1.8, 'South Korea': 1.6, 'India': 1.4, 'Australia': 1.2, 'Italy': 0.9, 'Denmark': 0.8, 'Spain': 0.7, 'Hong Kong': 0.4, 'Other': 4.4},
            'VWCE.DE': {'United States': 63.0, 'Japan': 5.8, 'United Kingdom': 3.8, 'Taiwan': 3.2, 'France': 3.0, 'Canada': 3.0, 'Switzerland': 2.8, 'Germany': 2.2, 'Netherlands': 1.8, 'South Korea': 1.6, 'India': 1.4, 'Australia': 1.2, 'Italy': 0.9, 'Denmark': 0.8, 'Spain': 0.7, 'Hong Kong': 0.4, 'Other': 4.4},
            'IWDA.L': {'United States': 71.0, 'Japan': 5.8, 'United Kingdom': 3.8, 'Taiwan': 3.1, 'France': 2.9, 'Canada': 2.7, 'Switzerland': 2.6, 'Germany': 2.1, 'Netherlands': 1.7, 'South Korea': 1.5, 'Australia': 1.2, 'Other': 1.6},
            
            # Europe ETFs
            'EXSA.DE': {'Germany': 20.0, 'France': 15.0, 'United Kingdom': 12.0, 'Switzerland': 10.0, 'Netherlands': 8.0, 'Italy': 7.0, 'Spain': 6.0, 'Denmark': 5.0, 'Other': 17.0},
            
            # Emerging Markets
            'EMIM.L': {'China': 30.0, 'Taiwan': 15.0, 'India': 12.0, 'South Korea': 10.0, 'Saudi Arabia': 5.0, 'Brazil': 5.0, 'South Africa': 4.0, 'Other': 19.0},
            'XMME.L': {'China': 30.0, 'Taiwan': 15.0, 'India': 12.0, 'South Korea': 10.0, 'Saudi Arabia': 5.0, 'Brazil': 5.0, 'South Africa': 4.0, 'Other': 19.0},
            
            # Specific Country/Region ETFs
            'GREK': {'Greece': 100.0},
            'SPOL.L': {'Poland': 100.0},
            
            # Sector ETFs (maintain sector focus but show geographic exposure)
            'AIAI.MI': {'United States': 70.0, 'Taiwan': 10.0, 'Netherlands': 8.0, 'South Korea': 5.0, 'Other': 7.0},  # AI ETF
            'AIAI.SW': {'United States': 70.0, 'Taiwan': 10.0, 'Netherlands': 8.0, 'South Korea': 5.0, 'Other': 7.0},  # AI ETF
            'SEMI.AS': {'Taiwan': 35.0, 'United States': 30.0, 'Netherlands': 15.0, 'South Korea': 10.0, 'Other': 10.0},  # Semiconductor
            'WISE': {'United States': 80.0, 'Taiwan': 8.0, 'Netherlands': 5.0, 'Other': 7.0},  # AI/Tech
            'INRG.L': {'United States': 40.0, 'China': 20.0, 'Denmark': 8.0, 'Spain': 6.0, 'Germany': 6.0, 'Other': 20.0},  # Clean Energy
            'HEAL.L': {'United States': 60.0, 'Switzerland': 10.0, 'Denmark': 8.0, 'United Kingdom': 7.0, 'Other': 15.0},  # Healthcare Innovation
            'XDWT.L': {'United States': 85.0, 'Taiwan': 8.0, 'Netherlands': 3.0, 'Other': 4.0},  # World Tech
            
            # Bond ETFs (show issuer countries)
            'IGLO.L': {'United States': 40.0, 'Japan': 20.0, 'Germany': 15.0, 'United Kingdom': 10.0, 'Other': 15.0},  # Global Gov Bonds
            'IEAC.L': {'Germany': 25.0, 'France': 20.0, 'Netherlands': 12.0, 'Italy': 10.0, 'Spain': 8.0, 'Other': 25.0},  # Euro Corp Bonds
            
            # Thematic/Specialty ETFs
            'EUAD': {'France': 35.0, 'United States': 25.0, 'Germany': 15.0, 'United Kingdom': 10.0, 'Other': 15.0},  # Aerospace & Defense
            'BNXG.DE': {'United States': 60.0, 'Canada': 15.0, 'Japan': 8.0, 'South Korea': 5.0, 'Other': 12.0},  # Blockchain
            'CEM.PA': {'Germany': 20.0, 'France': 18.0, 'Italy': 15.0, 'United Kingdom': 12.0, 'Other': 35.0},  # Europe Small Cap
        }
        
        # Check if we have specific mapping for this ETF
        if symbol in etf_geographic_mapping:
            return etf_geographic_mapping[symbol]
        
        # Fallback to pattern-based analysis
        if any(pattern in name for pattern in ['S&P 500', 'US EQUITY', 'AMERICA']):
            return {'United States': 100.0}
        elif any(pattern in name for pattern in ['WORLD', 'GLOBAL', 'ALL-WORLD']):
            return {'United States': 65.0, 'Japan': 6.0, 'United Kingdom': 4.0, 'Other': 25.0}
        elif any(pattern in name for pattern in ['EUROPE', 'STOXX EUROPE']):
            return {'Germany': 20.0, 'France': 15.0, 'United Kingdom': 12.0, 'Other': 53.0}
        elif any(pattern in name for pattern in ['EMERGING', 'EM IMI']):
            return {'China': 30.0, 'Taiwan': 15.0, 'India': 12.0, 'Other': 43.0}
        elif 'CHINA' in name:
            return {'China': 100.0}
        elif 'JAPAN' in name:
            return {'Japan': 100.0}
        elif any(pattern in name for pattern in ['GREECE', 'GREK']):
            return {'Greece': 100.0}
        elif 'POLAND' in name:
            return {'Poland': 100.0}
        
        # Default: use ETF domicile/exchange country as fallback
        return {stock_info['country']: 100.0}

    def export_for_superset(self, portfolio_name="Portfolio"):
        """Export portfolio data for Apache Superset dashboard."""
        if not hasattr(self, 'stock_info'):
            print("No data available. Run fetch_data() first.")
            return
        
        # Create database and tables for Superset
        db_path = f"portfolio_data.db"
        conn = sqlite3.connect(db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                portfolio_name TEXT,
                symbol TEXT,
                name TEXT,
                weight REAL,
                country TEXT,
                sector TEXT,
                industry TEXT,
                is_etf BOOLEAN,
                date_analyzed DATE,
                PRIMARY KEY (portfolio_name, symbol)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS country_allocation (
                portfolio_name TEXT,
                country TEXT,
                weight REAL,
                date_analyzed DATE,
                PRIMARY KEY (portfolio_name, country)
            )
        ''')
        
        # Delete existing data for this portfolio before inserting new data
        conn.execute('DELETE FROM portfolio_holdings WHERE portfolio_name = ?', (portfolio_name,))
        conn.execute('DELETE FROM country_allocation WHERE portfolio_name = ?', (portfolio_name,))
        
        # Insert holdings data
        current_date = datetime.now().strftime('%Y-%m-%d')
        holdings_data = []
        
        for i, stock in enumerate(self.stocks):
            stock_info = self.stock_info[stock]
            holdings_data.append((
                portfolio_name,
                stock,
                stock_info['name'],
                self.weights[i],
                stock_info['country'],
                stock_info['sector'],
                stock_info['industry'],
                stock_info['is_etf'],
                current_date
            ))
        
        conn.executemany('''
            INSERT OR REPLACE INTO portfolio_holdings 
            (portfolio_name, symbol, name, weight, country, sector, industry, is_etf, date_analyzed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', holdings_data)
        
        # Insert country allocation data
        country_weights = self.analyze_geographic_allocation(portfolio_name)
        country_data = [(portfolio_name, country, weight, current_date) 
                       for country, weight in country_weights.items()]
        
        conn.executemany('''
            INSERT OR REPLACE INTO country_allocation 
            (portfolio_name, country, weight, date_analyzed)
            VALUES (?, ?, ?, ?)
        ''', country_data)
        
        conn.commit()
        conn.close()
        
        print(f"\n💾 Data exported to {db_path} for Apache Superset")
        print("📊 Tables created: portfolio_holdings, country_allocation")
        
        # Create Superset configuration file
        self._create_superset_config()
        
        return db_path

    def _create_superset_config(self):
        """Create Apache Superset configuration files and dashboard definition."""
        
        # Create dashboard configuration
        dashboard_config = {
            "dashboard_title": "Portfolio Geographic Analysis",
            "description": "Interactive dashboard showing portfolio allocation by country",
            "charts": [
                {
                    "chart_type": "world_map",
                    "title": "Portfolio Geographic Distribution",
                    "table": "country_allocation",
                    "metrics": ["weight"],
                    "filters": ["portfolio_name"]
                },
                {
                    "chart_type": "pie",
                    "title": "Country Allocation (Pie Chart)",
                    "table": "country_allocation", 
                    "metrics": ["weight"],
                    "groupby": ["country"],
                    "filters": ["portfolio_name"]
                },
                {
                    "chart_type": "bar",
                    "title": "Country Allocation (Bar Chart)",
                    "table": "country_allocation",
                    "metrics": ["weight"], 
                    "groupby": ["country"],
                    "filters": ["portfolio_name"]
                }
            ]
        }
        
        with open('superset_dashboard_config.json', 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        print("📋 Superset dashboard config saved as 'superset_dashboard_config.json'")

    def generate_quantstats_report(self, save_html=False):
        """Generate comprehensive QuantStats report."""
        print("\n" + "="*60)
        print("GENERATING QUANTSTATS ANALYSIS")
        print("="*60)
        if save_html:
            qs.reports.html(self.portfolio_returns, 
                            benchmark=self.benchmark_returns,
                            output='portfolio_report.html',
                            title='Portfolio vs S&P 500 Analysis')
            print("📄 Full HTML report saved as 'portfolio_report.html'")
        print("\n📊 PORTFOLIO PERFORMANCE METRICS:")
        print("-" * 40)
        print("\n📊 FULL QUANTSTATS METRICS TABLE:")
        qs.reports.metrics(self.portfolio_returns, 
                          benchmark=self.benchmark_returns,
                          display=True)

    def plot_daily_cumulative_returns(self):
        """Plot daily cumulative returns of the portfolio and S&P 500 benchmark."""
        if not hasattr(self, 'portfolio_returns') or not hasattr(self, 'benchmark_returns'):
            print("Portfolio or benchmark returns not calculated yet. Calculating now...")
            self.calculate_portfolio_returns()
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod() - 1
        cumulative_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_portfolio.index, cumulative_portfolio * 100, label='Portfolio', color='blue')
        plt.plot(cumulative_benchmark.index, cumulative_benchmark * 100, label='S&P 500', color='red', alpha=0.7)
        plt.title(f'Daily Cumulative Returns: Portfolio vs S&P 500\n({self.start_date} to {self.end_date})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fix overlapping x-axis labels
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        # Add data period information
        trading_days = len(cumulative_portfolio)
        plt.text(0.02, 0.98, f'Trading Days: {trading_days}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()

    def create_comparison_plots(self):
        """Create comparison visualizations."""
        plt.style.use('default')
        qs.extend_pandas()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Portfolio vs S&P 500 Analysis (QuantStats)\n{self.start_date} to {self.end_date}', fontsize=16, fontweight='bold')
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod() - 1
        cumulative_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        
        ax1.plot(cumulative_portfolio.index, cumulative_portfolio * 100, 
                label='Portfolio', linewidth=2, color='blue')
        ax1.plot(cumulative_benchmark.index, cumulative_benchmark * 100, 
                label='S&P 500', linewidth=2, color='red', alpha=0.7)
        ax1.set_title(f'Cumulative Returns\n({self.start_date} to {self.end_date})')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        # 2. Rolling Sharpe Ratio
        ax2 = axes[0, 1]
        rolling_sharpe_port = qs.stats.rolling_sharpe(self.portfolio_returns)
        rolling_sharpe_bench = qs.stats.rolling_sharpe(self.benchmark_returns)
        ax2.plot(rolling_sharpe_port.index, rolling_sharpe_port, 
                label='Portfolio', color='blue')
        ax2.plot(rolling_sharpe_bench.index, rolling_sharpe_bench, 
                label='S&P 500', color='red', alpha=0.7)
        ax2.set_title('Rolling 6-Month Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        # 3. Drawdown Comparison
        ax3 = axes[1, 0]
        dd_portfolio = qs.stats.to_drawdown_series(self.portfolio_returns)
        dd_benchmark = qs.stats.to_drawdown_series(self.benchmark_returns)
        ax3.fill_between(dd_portfolio.index, dd_portfolio * 100, 0, 
                        alpha=0.7, color='blue', label='Portfolio')
        ax3.fill_between(dd_benchmark.index, dd_benchmark * 100, 0, 
                        alpha=0.5, color='red', label='S&P 500')
        ax3.set_title('Drawdown Comparison')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        # 4. Monthly Returns Heatmap (Portfolio only)
        ax4 = axes[1, 1]
        monthly_returns = qs.stats.monthly_returns(self.portfolio_returns)
        monthly_rets = self.portfolio_returns.resample('M').apply(qs.stats.comp)
        monthly_rets.plot(kind='bar', ax=ax4, color='blue', alpha=0.7)
        ax4.set_title('Portfolio Monthly Returns')
        ax4.set_ylabel('Monthly Return')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.tight_layout()

    def quick_analysis(self):
        """Quick analysis using QuantStats built-in functions."""
        print("\n🚀 QUICK QUANTSTATS ANALYSIS")
        print("=" * 50)
        qs.reports.metrics(self.portfolio_returns, 
                          benchmark=self.benchmark_returns, 
                          display=True)

    def run_analysis(self, generate_html=False):
        """Run the complete analysis."""
        try:
            self.fetch_data()
            if self.stock_data.empty:
                print("No data available for analysis")
                return
            self.calculate_portfolio_returns()
            self.quick_analysis()
            self.generate_quantstats_report(save_html=generate_html)
            self.create_comparison_plots()
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Define four portfolios
    for s in (0,1):
        if s == 0:
            portfolio1  = {
                'CSSPX.MI': 0.45,
                'SOXX': 0.2,
                'SEMI.AS' :0.15,
                'ASML' :0.1,
                'WISE': 0.1
            }
            portfolio2  = {
                'CSSPX.MI': 0.3,
                'VWRD.L': 0.3,
                'NVDA': 0.1,
                'ENPH': 0.1,
                'MRNA': 0.1,
                'NET': 0.1
            }
            portfolio3 = {
                'CSSPX.MI': 0.5,            
                'AIAI.MI': 0.2,
                'SGLE.MI': 0.15,
                'CEM.PA': 0.1,
                'NOW': 0.05
            }
            portfolio4 = {
                'IWDA.L': 0.5,
                'EMIM.L': 0.2,
                'IGLO.L': 0.2,
                'ASML': 0.02,
                'MOH.MU': 0.02,
                'MSFT': 0.02,
                'AAPL': 0.02,
                'NESM.SG': 0.02
            }
            portfolio5 = {
                'EUAD': 0.25,
                'GREK': 0.2,
                'AIAI.SW': 0.15,
                'BNXG.DE': 0.15,
                'SPOL.L': 0.1,
                'NVDA': 0.0375,
                'AMD': 0.0375,
                'GOOG': 0.0375,
                'TSMN.MX': 0.0375
            }
        else:
            portfolio1  = {
                'IWDA.L': 0.2,
                'XMME.L': 0.2,
                'EQAC.MI' :0.1,
                'HEAL.L' :0.1,
                'MSFT': 0.05,
                'NESM.SG': 0.05
            }
            portfolio2  = {
                'VWCE.DE': 0.5,
                'CSSPX.MI': 0.25,
                'MSFT': 0.05,
                'NVO': 0.05,
                'ASML': 0.05,
                'OR': 0.05,
                'V': 0.05
                
            }
            portfolio3 = {
                'IWDA.L': 0.24,            
                'VWRD.L': 0.24,
                'INRG.L': 0.12,
                'MSFT': 0.075,
                'ASML': 0.06,
                'NESM.SG': 0.06,
                'ROG.SW': 0.06,
                'NEE': 0.045,
                'EMIM.L': 0.1
            }
            portfolio4 = {
                'VWCE.DE': 0.7,
                'EMIM.L': 0.1,
                'IEAC.L': 0.2
                
            }
            portfolio5 = {
                'IWDA.L': 0.35,
                'EMIM.L': 0.15,
                'EXSA.DE': 0.1,
                'VUSA.L': 0.1,
                'XDWT.L': 0.1,
                'NESM.SG': 0.05,
                'NOVO-B.CO': 0.05,
                'MSFT': 0.05,
                'ASML': 0.05 
            }
        # Analyze all portfolios
        portfolios = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5]
        labels = ['ChatGPT', 'Gemini', 'Mistral.ai', 'DeepSeek', 'Perplexity AI']
        colors = ['blue', 'green', 'orange', 'purple', 'cyan']
        
        analyzers = []
        term_type = "Short Term" if s == 0 else "Long Term"
        for i, p in enumerate(portfolios):
            analyzer = PortfolioAnalyzer(
                portfolio_stocks=p,
                start_date=['2025-09-10', '2025-09-25'][s],
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            analyzer.fetch_data()
            analyzer.calculate_portfolio_returns()
            # Create distinct portfolio name for short/long term
            portfolio_full_name = f"{labels[i]} {term_type}"
            
            # Print portfolio header
            print(f"\n{'='*80}")
            print(f"ANALYZING PORTFOLIO: {portfolio_full_name}")
            print(f"{'='*80}")
            print(f"Portfolio: {p}")
            print(f"Analysis period: {analyzer.start_date} to {analyzer.end_date}")
            print("Fetching data...")
            
            analyzer.analyze_sector_allocation(portfolio_full_name)  # Add sector analysis
            analyzer.analyze_geographic_allocation(portfolio_full_name)  # Add geographic analysis
            analyzer.export_for_superset(portfolio_full_name)  # Export data for Superset dashboard
            analyzers.append(analyzer)
        # Plot all portfolios and benchmark on the same chart
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative returns for all portfolios
        portfolio_cumulatives = []
        
        for i, analyzer in enumerate(analyzers):
            cumulative = (1 + analyzer.portfolio_returns).cumprod() - 1
            # Add September 10th with 0% value
            cumulative.loc[pd.Timestamp(['2025-09-10', '2025-09-25'][s])] = 0.0
            cumulative = cumulative.sort_index()  # Sort to put Sept 10th in correct chronological position
            portfolio_cumulatives.append(cumulative)
            plt.plot(cumulative.index, cumulative * 100, label=labels[i], color=colors[i])
        
        # Calculate benchmark cumulative returns
        cumulative_benchmark_normalized = (1 + analyzers[0].benchmark_returns).cumprod() - 1
        # Add September 10th with 0% value for benchmark
        cumulative_benchmark_normalized.loc[pd.Timestamp(['2025-09-10', '2025-09-25'][s])] = 0.0
        cumulative_benchmark_normalized = cumulative_benchmark_normalized.sort_index()
        plt.plot(cumulative_benchmark_normalized.index, cumulative_benchmark_normalized * 100, label='S&P 500', color='red', alpha=0.7)
        if s == 0:
            plt.title(f'Daily Cumulative Returns: 5 Short Term Portfolios vs S&P 500\n({analyzers[0].start_date} to {analyzers[0].end_date})')
        else:
            plt.title(f'Daily Cumulative Returns: 5 Long Term Portfolios vs S&P 500\n({analyzers[0].start_date} to {analyzers[0].end_date})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)') 
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fix overlapping x-axis labels
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        # Add data period information
        trading_days = len(cumulative_benchmark_normalized)
        plt.text(0.02, 0.98, f'Trading Days: {trading_days}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
    plt.show(block=True)   # Block only for last graph
    input("Press Enter to close the plots...")