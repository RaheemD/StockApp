import pandas as pd
import numpy as np
import yfinance as yf

def calculate_technical_indicators(data, indicators):
    """
    Calculate various technical indicators for the stock data.
    
    Parameters:
    - data: Pandas DataFrame with OHLC data
    - indicators: List of indicators to calculate
    
    Returns:
    - DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Calculate Simple Moving Averages (SMA)
    if "SMA" in indicators:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Exponential Moving Averages (EMA)
    if "EMA" in indicators:
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Calculate Relative Strength Index (RSI)
    if "RSI" in indicators:
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    if "MACD" in indicators:
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate Bollinger Bands
    if "Bollinger Bands" in indicators:
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_StdDev'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
    
    return df

def get_company_info(symbol):
    """
    Get detailed company information for a given stock symbol.
    
    Parameters:
    - symbol: Stock ticker symbol
    
    Returns:
    - Dictionary with company information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return {}

def format_large_number(num):
    """
    Format large numbers for better readability.
    
    Parameters:
    - num: Number to format
    
    Returns:
    - Formatted string (e.g., "1.2B" for 1,200,000,000)
    """
    if not isinstance(num, (int, float)):
        return "N/A"
    
    if num >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num / 1_000:.2f}K"
    else:
        return f"${num:.2f}"

def calculate_returns(data):
    """
    Calculate various return metrics for the stock data.
    
    Parameters:
    - data: Pandas DataFrame with OHLC data
    
    Returns:
    - DataFrame with return metrics
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Create a new DataFrame for returns
        returns = pd.DataFrame(index=df.index)
        
        # Calculate daily returns (percentage change)
        returns['Daily Return'] = df['Close'].pct_change()
        
        # Replace inf and NaN values with 0
        returns['Daily Return'] = returns['Daily Return'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate cumulative returns
        returns['Cumulative Return'] = (1 + returns['Daily Return']).cumprod() - 1
        returns['Cumulative Return'] = returns['Cumulative Return'] * 100  # Convert to percentage
        
        # Ensure all values are properly formatted as numeric
        returns = returns.apply(pd.to_numeric, errors='coerce')
        
        # Fill any remaining NaN values with 0
        returns = returns.fillna(0)
        
        return returns
    except Exception as e:
        print(f"Error calculating returns: {e}")
        # Return an empty DataFrame with the same structure if there's an error
        empty_returns = pd.DataFrame(index=[data.index[0]] if len(data) > 0 else [])
        empty_returns['Daily Return'] = 0.0
        empty_returns['Cumulative Return'] = 0.0
        return empty_returns
