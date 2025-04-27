# Stock Analysis Dashboard

A dark-themed, web-based stock analysis tool using Streamlit that pulls live data from Yahoo Finance and displays interactive visualizations.

## Features

- Summary table of essential financial metrics for popular companies
- Interactive chart of historical stock price trends
- Additional charts for key financial indicators
- Responsive design with smooth transitions, optimized for mobile use
- Dark mode UI with minimalist design that emphasizes data clarity and readability
- Database persistence for stock data caching and user preferences
- Favorite stocks management for quick access
- Save and reload analysis configurations

## Technical Features

- Real-time stock data from Yahoo Finance
- Interactive historical stock price charts using Plotly
- Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- Performance and volume analysis
- Financial metrics and ratios
- Analyst recommendations
- PostgreSQL database for data persistence
- Stock data caching for improved performance

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required packages:
   ```
   pip install streamlit yfinance pandas plotly numpy
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Select a stock symbol from the dropdown, favorites list, or enter a custom symbol
2. Choose a time period for the historical data
3. Select chart type (Candlestick, Line, or Area)
4. Choose technical indicators to display
5. Explore different analysis tabs for more insights
6. Save favorites stocks for quick access later
7. Save and reload analysis configurations with your preferred settings

## Data Sources

All stock data is fetched in real-time from Yahoo Finance using the `yfinance` package.

## Disclaimer

This tool is for educational purposes only. It is not intended to provide investment advice. Data may be delayed. Always conduct thorough research before making investment decisions.
