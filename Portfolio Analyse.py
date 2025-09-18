import pandas as pd
import yfinance as yf
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
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
                portfolio_data[stock] = data['Close']
                print(f"✓ {name} ({stock}) data fetched")
            except Exception as e:
                print(f"✗ Error fetching {stock}: {e}")
        try:
            sp500 = yf.Ticker("SPY")
            sp500_data = sp500.history(start=self.start_date, end=self.end_date)
            self.benchmark_data = sp500_data['Close']
            print("✓ S&P 500 (SPY) data fetched")
        except Exception as e:
            print(f"✗ Error fetching S&P 500: {e}")
        # Remove timezone info from each series before combining
        for stock in portfolio_data:
            if portfolio_data[stock].index.tz is not None:
                portfolio_data[stock].index = portfolio_data[stock].index.tz_localize(None)
        self.stock_data = pd.DataFrame(portfolio_data)
        self.stock_data = self.stock_data.dropna()
        if self.stock_data.empty:
            print("\n[WARNING] No overlapping data for all tickers in the selected date range. Try a more recent start_date or check ticker data on Yahoo Finance.")
        self.stock_data.index = self.stock_data.index.tz_localize(None)
        self.benchmark_data.index = self.benchmark_data.index.tz_localize(None)
        return self.stock_data, self.benchmark_data

    def calculate_portfolio_returns(self):
        """Calculate portfolio returns using weights."""
        stock_returns = self.stock_data.pct_change().dropna()
        self.portfolio_returns = (stock_returns * self.weights).sum(axis=1)
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()
        common_dates = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        self.portfolio_returns = self.portfolio_returns.loc[common_dates]
        self.benchmark_returns = self.benchmark_returns.loc[common_dates]
        return self.portfolio_returns, self.benchmark_returns

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
        plt.title('Daily Cumulative Returns: Portfolio vs S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def create_comparison_plots(self):
        """Create comparison visualizations."""
        plt.style.use('default')
        qs.extend_pandas()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio vs S&P 500 Analysis (QuantStats)', fontsize=16, fontweight='bold')
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod() - 1
        cumulative_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        ax1.plot(cumulative_portfolio.index, cumulative_portfolio * 100, 
                label='Portfolio', linewidth=2, color='blue')
        ax1.plot(cumulative_benchmark.index, cumulative_benchmark * 100, 
                label='S&P 500', linewidth=2, color='red', alpha=0.7)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
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
    portfolio1  = {
        'CSSPX.MI': 0.45,
        'SOXX': 0.2,
        'SEMI.AS' :0.15,
        'ASML.AS' :0.1,
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
        'NVDA': 0.15,
        'CEM.PA': 0.1,
        'NOW': 0.05

    }
    portfolio4 = {
        'IWDA.L': 0.5,
        'EMIM.L': 0.2,
        'IGLO.L': 0.2,
        'ASML.AS': 0.02,
        'MC.PA': 0.02,
        'MSFT': 0.02,
        'AAPL': 0.02,
        'NESN.SW': 0.02
        
    }
    # Analyze all portfolios
    analyzers = []
    for p in [portfolio1, portfolio2, portfolio3, portfolio4]:
        analyzer = PortfolioAnalyzer(
            portfolio_stocks=p,
            start_date='2025-09-10',
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        analyzer.fetch_data()
        analyzer.calculate_portfolio_returns()
        analyzers.append(analyzer)
    # Plot all portfolios and benchmark on the same chart
    import matplotlib.pyplot as plt
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['ChatGPT', 'Gemini', 'Mistral.ai ', 'DeepSeek ']
    plt.figure(figsize=(12, 6))
    for i, analyzer in enumerate(analyzers):
        cumulative = (1 + analyzer.portfolio_returns).cumprod() - 1
        plt.plot(cumulative.index, cumulative * 100, label=labels[i], color=colors[i])
    cumulative_benchmark = (1 + analyzers[0].benchmark_returns).cumprod() - 1
    plt.plot(cumulative_benchmark.index, cumulative_benchmark * 100, label='S&P 500', color='red', alpha=0.7)
    plt.title('Daily Cumulative Returns: 4 Portfolios vs S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)
    input("Press Enter to close the plots...")