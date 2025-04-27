import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import uuid
from datetime import datetime, timedelta
from utils import (
    calculate_technical_indicators,
    get_company_info,
    format_large_number,
    calculate_returns
)
import db

# Initialize the database if needed
db.init_db()

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📈 Stock Analysis Dashboard")
st.markdown("Analyze stock performance and financial metrics using real-time data from Yahoo Finance")

# Initialize session state for user management
if 'user_id' not in st.session_state:
    # Generate a random user ID for the session (In a real app, this would be from authentication)
    st.session_state.user_id = str(uuid.uuid4())

# Function to load stock data - now with database caching
def load_stock_data(symbol, period):
    """
    Load stock data for the given symbol and period, with database caching.
    
    Parameters:
    - symbol: Stock ticker symbol
    - period: Time period string (e.g., '1mo', '1y')
    
    Returns:
    - DataFrame with stock data
    """
    try:
        # First try to get from cache
        cached_data = db.get_cached_price_data(symbol, period)
        
        # If not in cache or data is old, fetch from Yahoo Finance
        if cached_data is None:
            # Convert period string to yfinance format
            period_map = {
                '1 Week': '1wk', 
                '1 Month': '1mo', 
                '3 Months': '3mo',
                '6 Months': '6mo', 
                '1 Year': '1y', 
                '2 Years': '2y', 
                '5 Years': '5y'
            }
            yf_period = period_map.get(period, '1y')
            
            # Download stock data from Yahoo Finance
            data = yf.download(symbol, period=yf_period)
            
            if not data.empty:
                try:
                    # Get company info
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    
                    # Ensure data is properly formatted
                    data.index = pd.to_datetime(data.index)
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    for col in required_columns:
                        if col not in data.columns:
                            data[col] = None
                    
                    # Save to database
                    stock_id = db.add_or_update_stock(symbol, info)
                    if stock_id:
                        db.add_stock_prices(stock_id, data)
                except Exception as db_error:
                    st.warning(f"Could not save stock data to database: {db_error}")
            
            return data
        
        return cached_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # Tabs for selection methods
    symbol_tabs = st.tabs(["Browse", "Favorites", "Custom"])
    
    with symbol_tabs[0]:
        # Stock symbol input from default list
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        selected_symbol = st.selectbox(
            "Select a stock symbol:",
            options=default_symbols,
            index=0
        )
    
    with symbol_tabs[1]:
        # Fetch user's favorite stocks from database
        favorites = db.get_user_favorites(st.session_state.user_id)
        favorite_symbols = []
        
        if favorites:
            # Create a dictionary mapping of symbol to company name for the dropdown
            favorite_options = {}
            for fav in favorites:
                favorite_symbols.append(fav['symbol'])
                favorite_options[fav['symbol']] = f"{fav['symbol']} - {fav['company_name']}"
            
            # Create a dropdown with the user's favorites
            selected_favorite = st.selectbox(
                "Select from your favorites:",
                options=favorite_symbols,
                format_func=lambda x: favorite_options.get(x, x),
                index=0 if favorite_symbols else None
            )
            
            if selected_favorite:
                selected_symbol = selected_favorite
                
            # Show remove button for the selected favorite
            if st.button("Remove from Favorites", key="remove_fav"):
                success = db.remove_from_favorites(st.session_state.user_id, selected_symbol)
                if success:
                    st.success(f"Removed {selected_symbol} from favorites")
                    st.rerun()
                else:
                    st.error("Failed to remove from favorites")
        else:
            st.info("You don't have any favorites yet. Add some from the analyzer!")
    
    with symbol_tabs[2]:
        # Custom symbol input
        custom_symbol = st.text_input("Enter a custom symbol:", "")
        if custom_symbol:
            selected_symbol = custom_symbol.upper()
    
    # Add to favorites button
    if st.button("➕ Add Current Stock to Favorites"):
        # Check if already in favorites
        favorites = db.get_user_favorites(st.session_state.user_id)
        already_favorite = False
        for fav in favorites:
            if fav['symbol'] == selected_symbol:
                already_favorite = True
                break
        
        if already_favorite:
            st.sidebar.info(f"{selected_symbol} is already in your favorites")
        else:
            success = db.add_to_favorites(st.session_state.user_id, selected_symbol)
            if success:
                st.sidebar.success(f"Added {selected_symbol} to favorites")
            else:
                st.sidebar.error("Failed to add to favorites")
    
    # Time period selection
    time_periods = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }
    selected_period = st.selectbox("Select time period:", list(time_periods.keys()), index=4)
    days = time_periods[selected_period]
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select chart type:",
        ["Candlestick", "Line", "Area"],
        index=0
    )
    
    # Technical indicators selection
    tech_indicators = st.multiselect(
        "Select technical indicators:",
        ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands"],
        default=["SMA", "EMA"]
    )
    
    # Save analysis section
    st.divider()
    st.subheader("Save Analysis")
    
    # Get saved analyses
    saved_analyses = db.get_saved_analyses(st.session_state.user_id)
    
    # Tabs for Save/Load
    save_tabs = st.tabs(["Save Current", "Load Saved"])
    
    with save_tabs[0]:
        analysis_name = st.text_input("Analysis name:", placeholder="My AAPL Analysis")
        notes = st.text_area("Notes:", placeholder="Optional notes about this analysis...")
        
        if st.button("Save Current Analysis"):
            if analysis_name:
                success = db.save_analysis(
                    user_id=st.session_state.user_id,
                    stock_symbol=selected_symbol,
                    analysis_name=analysis_name,
                    period=selected_period,
                    interval="1d",  # Default daily interval
                    indicators=tech_indicators,
                    chart_type=chart_type,
                    notes=notes
                )
                
                if success:
                    st.success(f"Analysis '{analysis_name}' saved successfully!")
                else:
                    st.error("Failed to save analysis. Please try again.")
            else:
                st.warning("Please provide an analysis name")
    
    with save_tabs[1]:
        if saved_analyses:
            # Create a dictionary for the dropdown
            analysis_options = {}
            for analysis in saved_analyses:
                key = f"{analysis['id']}"
                value = f"{analysis['name']} ({analysis['symbol']})"
                analysis_options[key] = value
            
            # Dropdown for saved analyses
            selected_analysis_id = st.selectbox(
                "Select a saved analysis:",
                options=list(analysis_options.keys()),
                format_func=lambda x: analysis_options.get(x, x),
                index=0
            )
            
            # Find the selected analysis
            selected_analysis = None
            for analysis in saved_analyses:
                if str(analysis['id']) == selected_analysis_id:
                    selected_analysis = analysis
                    break
            
            if selected_analysis:
                st.write(f"**Symbol**: {selected_analysis['symbol']}")
                st.write(f"**Period**: {selected_analysis['period']}")
                st.write(f"**Chart type**: {selected_analysis['chart_type']}")
                st.write(f"**Indicators**: {', '.join(selected_analysis['indicators'])}")
                
                if selected_analysis['notes']:
                    st.write(f"**Notes**: {selected_analysis['notes']}")
                
                # Load button
                if st.button("Load This Analysis"):
                    # Update session state or variables
                    selected_symbol = selected_analysis['symbol']
                    for i, period in enumerate(time_periods.keys()):
                        if period == selected_analysis['period']:
                            selected_period_index = i
                            break
                    
                    # Force a rerun to apply changes
                    st.rerun()
        else:
            st.info("You haven't saved any analyses yet.")

# This function is defined above, removed duplicate
# Using the database-backed implementation instead

# Main content
try:
    # Display loading indicator while fetching data
    with st.spinner(f"Loading data for {selected_symbol}..."):
        # Load stock data
        data = load_stock_data(selected_symbol, days)
        
        if data is None or data.empty:
            st.error(f"No data available for symbol {selected_symbol}. Please check the symbol and try again.")
            st.stop()
        
        # Get company information
        info = get_company_info(selected_symbol)
        
        # Create layout with two columns for key metrics
        col1, col2 = st.columns(2)
        
        # Display company name and basic info
        with col1:
            if 'longName' in info:
                st.subheader(info['longName'])
                if 'sector' in info and 'industry' in info:
                    st.text(f"Sector: {info['sector']} | Industry: {info['industry']}")
            else:
                st.subheader(selected_symbol)
        
        with col2:
            if 'website' in info:
                st.markdown(f"[Company Website]({info['website']})")
            if 'longBusinessSummary' in info:
                with st.expander("Business Summary"):
                    st.write(info['longBusinessSummary'])
    
        # Current stock price and market metrics
        price_data = yf.Ticker(selected_symbol).history(period='1d')
        if not price_data.empty:
            current_price = price_data['Close'].iloc[-1]
            prev_close = price_data['Open'].iloc[-1]
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
            
            # Display key financial metrics if available
            with metric_cols[1]:
                market_cap = info.get('marketCap', None)
                st.metric("Market Cap", format_large_number(market_cap) if market_cap else "N/A")
            
            with metric_cols[2]:
                pe_ratio = info.get('trailingPE', None)
                st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
            
            with metric_cols[3]:
                dividend_yield = info.get('dividendYield', None)
                dividend_display = f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A"
                st.metric("Dividend Yield", dividend_display)
            
            with metric_cols[4]:
                volume = price_data['Volume'].iloc[-1]
                st.metric("Volume", format_large_number(volume) if volume else "N/A")
        
        # Main stock price chart
        st.subheader(f"{selected_symbol} Stock Price Chart")
        
        # Calculate technical indicators if data exists and user selected them
        if not data.empty and tech_indicators:
            data = calculate_technical_indicators(data, tech_indicators)
        
        # Create figure for stock price chart
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            )])
        elif chart_type == "Line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00FFFF', width=2)
            ))
        else:  # Area chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                fill='tozeroy',
                mode='lines',
                name='Close Price',
                line=dict(color='#00FFFF', width=2),
                fillcolor='rgba(0, 255, 255, 0.2)'
            ))
            
        # Add technical indicators to the chart
        if 'SMA' in tech_indicators and 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#FF8C00', width=1.5)
            ))
            
        if 'EMA' in tech_indicators and 'EMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['EMA_20'],
                mode='lines',
                name='EMA 20',
                line=dict(color='#FF00FF', width=1.5)
            ))
            
        if 'Bollinger Bands' in tech_indicators and 'BB_Upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(173, 255, 47, 0.7)', width=1),
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(173, 255, 47, 0.7)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 255, 47, 0.05)'
            ))
            
        # Update layout for dark theme
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis sections in tabs
        tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Volume Analysis", "Technical Indicators"])
        
        with tab1:
            try:
                # Calculate returns
                returns_data = calculate_returns(data)
                
                # Returns chart
                st.subheader("Returns Analysis")
                
                # Create a copy with numeric values only for plotting
                returns_plot_data = returns_data.copy()
                
                # Create the line chart
                returns_fig = px.line(
                    returns_plot_data,
                    x=returns_plot_data.index,
                    y='Cumulative Return',
                    title="Cumulative Return (%)"
                )
                
                returns_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=400
                )
                
                st.plotly_chart(returns_fig, use_container_width=True)
                
                # Performance metrics table
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        # Extract values as Python floats (not pandas Series) using iloc[0] for scalar values
                        daily_mean = returns_data['Daily Return'].mean()
                        if isinstance(daily_mean, pd.Series):
                            avg_daily = float(daily_mean.iloc[0] * 100)
                        else:
                            avg_daily = float(daily_mean * 100)
                            
                        daily_sum = returns_data['Daily Return'].sum()
                        if isinstance(daily_sum, pd.Series):
                            monthly_return = float(daily_sum.iloc[0] * 100)
                        else:
                            monthly_return = float(daily_sum * 100)
                            
                        cumulative = returns_data['Cumulative Return'].iloc[-1]
                        if isinstance(cumulative, pd.Series):
                            total_return = float(cumulative.iloc[0])
                        else:
                            total_return = float(cumulative)
                        
                        # Create a simple dictionary with string values
                        performance_data = {
                            "Metric": ["Daily Return (Avg)", "Monthly Return", "Total Return"],
                            "Value": [
                                f"{avg_daily:.2f}%",
                                f"{monthly_return:.2f}%",
                                f"{total_return:.2f}%"
                            ]
                        }
                        
                        # Create a fresh DataFrame from the dictionary
                        perf_df = pd.DataFrame(performance_data)
                        st.table(perf_df)
                    except Exception as e:
                        st.error(f"Could not display performance metrics: {e}")
            except Exception as e:
                st.error(f"Error in returns analysis: {e}")
                st.warning("Please try a different stock or time period.")
                
            with col2:
                if len(data) > 20:
                    try:
                        # Calculate risk metrics safely - convert to float values using iloc[0] where needed
                        vol_std = returns_data['Daily Return'].std()
                        if isinstance(vol_std, pd.Series):
                            volatility = float(vol_std.iloc[0] * 100)
                        else:
                            volatility = float(vol_std * 100)
                            
                        # Safely calculate max drawdown
                        drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
                        if isinstance(drawdown, pd.Series):
                            max_drawdown = float(drawdown.iloc[0] * 100)
                        else:
                            max_drawdown = float(drawdown * 100)
                        
                        # Calculate Sharpe ratio separately to avoid potential division issues
                        mean_ret = returns_data['Daily Return'].mean()
                        if isinstance(mean_ret, pd.Series):
                            mean_return = float(mean_ret.iloc[0])
                        else:
                            mean_return = float(mean_ret)
                            
                        std_ret = returns_data['Daily Return'].std()
                        if isinstance(std_ret, pd.Series):
                            std_return = float(std_ret.iloc[0])
                        else:
                            std_return = float(std_ret)
                        sharpe = 0.0
                        if std_return > 0:  # Prevent division by zero
                            sharpe = mean_return / std_return
                        
                        risk_data = {
                            "Metric": ["Volatility (Daily)", "Max Drawdown", "Sharpe Ratio (est.)"],
                            "Value": [
                                f"{volatility:.2f}%",
                                f"{max_drawdown:.2f}%",
                                f"{sharpe:.2f}"
                            ]
                        }
                        
                        # Create a fresh DataFrame from the dictionary
                        risk_df = pd.DataFrame(risk_data)
                        st.table(risk_df)
                    except Exception as e:
                        st.error(f"Could not display risk metrics: {e}")
        
        with tab2:
            # Volume analysis
            st.subheader("Trading Volume Analysis")
            
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker=dict(color='rgba(0, 255, 255, 0.7)')
            ))
            
            # Add moving average of volume
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            volume_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Volume_SMA'],
                name='20-day Volume SMA',
                line=dict(color='#FF00FF', width=2)
            ))
            
            volume_fig.update_layout(
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Volume",
                margin=dict(l=0, r=0, t=10, b=0),
                height=400
            )
            
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Volume statistics
            vol_stats = {
                "Metric": ["Average Volume", "Max Volume", "Min Volume", "Current vs Avg"],
                "Value": [
                    format_large_number(float(data['Volume'].mean().iloc[0]) if isinstance(data['Volume'].mean(), pd.Series) else float(data['Volume'].mean())),
                    format_large_number(float(data['Volume'].max().iloc[0]) if isinstance(data['Volume'].max(), pd.Series) else float(data['Volume'].max())),
                    format_large_number(float(data['Volume'].min().iloc[0]) if isinstance(data['Volume'].min(), pd.Series) else float(data['Volume'].min())),
                    f"{(float(data['Volume'].iloc[-1].iloc[0]) if isinstance(data['Volume'].iloc[-1], pd.Series) else float(data['Volume'].iloc[-1])) / (float(data['Volume'].mean().iloc[0]) if isinstance(data['Volume'].mean(), pd.Series) else float(data['Volume'].mean())) - 1 * 100:.2f}%"
                ]
            }
            
            st.table(pd.DataFrame(vol_stats))
            
        with tab3:
            # Technical indicators detailed view
            st.subheader("Technical Indicators")
            
            indicator_tabs = []
            if tech_indicators:
                indicator_tabs = st.tabs(tech_indicators)
            else:
                st.info("Please select technical indicators in the sidebar to display them here.")
            
            # Display each selected indicator in its own tab
            for i, indicator in enumerate(tech_indicators):
                with indicator_tabs[i]:
                    if indicator == "RSI" and "RSI" in data.columns:
                        rsi_fig = go.Figure()
                        rsi_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='#00FFFF', width=2)
                        ))
                        
                        # Add overbought and oversold lines
                        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        
                        rsi_fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Date",
                            yaxis_title="RSI Value",
                            yaxis=dict(range=[0, 100]),
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=400
                        )
                        
                        st.plotly_chart(rsi_fig, use_container_width=True)
                        st.write("""
                        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
                        - RSI above 70 generally indicates overbought conditions
                        - RSI below 30 generally indicates oversold conditions
                        """)
                    
                    elif indicator == "MACD" and "MACD" in data.columns:
                        macd_fig = go.Figure()
                        macd_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='#00FFFF', width=2)
                        ))
                        
                        macd_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MACD_Signal'],
                            mode='lines',
                            name='Signal Line',
                            line=dict(color='#FF00FF', width=1.5)
                        ))
                        
                        macd_fig.add_trace(go.Bar(
                            x=data.index,
                            y=data['MACD_Hist'],
                            name='Histogram',
                            marker=dict(color='rgba(255, 255, 255, 0.5)')
                        ))
                        
                        macd_fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Date",
                            yaxis_title="MACD Value",
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=400
                        )
                        
                        st.plotly_chart(macd_fig, use_container_width=True)
                        st.write("""
                        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator.
                        - MACD crossing above the signal line is typically considered bullish
                        - MACD crossing below the signal line is typically considered bearish
                        """)
                    
                    elif indicator == "Bollinger Bands" and "BB_Upper" in data.columns:
                        bb_fig = go.Figure()
                        bb_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='#FFFFFF', width=2)
                        ))
                        
                        bb_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['BB_Upper'],
                            mode='lines',
                            name='Upper Band',
                            line=dict(color='rgba(173, 255, 47, 0.7)', width=1),
                        ))
                        
                        bb_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['BB_Middle'],
                            mode='lines',
                            name='Middle Band (SMA 20)',
                            line=dict(color='rgba(255, 165, 0, 0.7)', width=1),
                        ))
                        
                        bb_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['BB_Lower'],
                            mode='lines',
                            name='Lower Band',
                            line=dict(color='rgba(173, 255, 47, 0.7)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(173, 255, 47, 0.05)'
                        ))
                        
                        bb_fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=400
                        )
                        
                        st.plotly_chart(bb_fig, use_container_width=True)
                        st.write("""
                        Bollinger Bands consist of a middle band (20-day SMA) with upper and lower bands at 2 standard deviations.
                        - Price touching the upper band may indicate overbought conditions
                        - Price touching the lower band may indicate oversold conditions
                        - Band contraction indicates low volatility, while expansion indicates high volatility
                        """)
                    
                    elif (indicator == "SMA" or indicator == "EMA") and ("SMA_20" in data.columns or "EMA_20" in data.columns):
                        ma_fig = go.Figure()
                        ma_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='#FFFFFF', width=2)
                        ))
                        
                        if indicator == "SMA" and "SMA_20" in data.columns:
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['SMA_20'],
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='#FF8C00', width=1.5)
                            ))
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['SMA_50'],
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='#00FFFF', width=1.5)
                            ))
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['SMA_200'],
                                mode='lines',
                                name='SMA 200',
                                line=dict(color='#FF00FF', width=1.5)
                            ))
                            
                        if indicator == "EMA" and "EMA_20" in data.columns:
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['EMA_20'],
                                mode='lines',
                                name='EMA 20',
                                line=dict(color='#FF8C00', width=1.5)
                            ))
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['EMA_50'],
                                mode='lines',
                                name='EMA 50',
                                line=dict(color='#00FFFF', width=1.5)
                            ))
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['EMA_200'],
                                mode='lines',
                                name='EMA 200',
                                line=dict(color='#FF00FF', width=1.5)
                            ))
                        
                        ma_fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=400
                        )
                        
                        st.plotly_chart(ma_fig, use_container_width=True)
                        if indicator == "SMA":
                            st.write("""
                            Simple Moving Averages (SMA) smooth price data to form a trend-following indicator.
                            - When short-term SMA crosses above long-term SMA, it's typically a bullish signal (golden cross)
                            - When short-term SMA crosses below long-term SMA, it's typically a bearish signal (death cross)
                            """)
                        else:
                            st.write("""
                            Exponential Moving Averages (EMA) give more weight to recent prices and react more quickly to price changes.
                            - EMA is typically more responsive to recent price changes than SMA
                            - The 50-day and 200-day EMAs are commonly used to identify major trend directions
                            """)
                            
        # Financial data section
        st.subheader("Financial Metrics")
        
        # Create tabs for different financial information
        fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Key Ratios", "Financial Growth", "Balance Sheet"])
        
        with fin_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Valuation metrics
                valuation_data = {
                    "Metric": [
                        "Market Cap", "Enterprise Value", 
                        "Trailing P/E", "Forward P/E", 
                        "PEG Ratio", "Price/Sales", 
                        "Price/Book", "Enterprise Value/Revenue"
                    ],
                    "Value": [
                        format_large_number(info.get('marketCap', 'N/A')),
                        format_large_number(info.get('enterpriseValue', 'N/A')),
                        f"{info.get('trailingPE', 'N/A')}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A',
                        f"{info.get('forwardPE', 'N/A')}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A',
                        f"{info.get('pegRatio', 'N/A')}" if isinstance(info.get('pegRatio'), (int, float)) else 'N/A',
                        f"{info.get('priceToSalesTrailing12Months', 'N/A')}" if isinstance(info.get('priceToSalesTrailing12Months'), (int, float)) else 'N/A',
                        f"{info.get('priceToBook', 'N/A')}" if isinstance(info.get('priceToBook'), (int, float)) else 'N/A',
                        f"{info.get('enterpriseToRevenue', 'N/A')}" if isinstance(info.get('enterpriseToRevenue'), (int, float)) else 'N/A'
                    ]
                }
                st.subheader("Valuation Metrics")
                st.table(pd.DataFrame(valuation_data))
            
            with col2:
                # Profitability metrics
                profitability_data = {
                    "Metric": [
                        "Profit Margin", "Operating Margin", 
                        "ROE", "ROA", 
                        "EBITDA Margin", "Gross Margin", 
                        "Free Cash Flow", "Operating Cash Flow"
                    ],
                    "Value": [
                        f"{info.get('profitMargins', 'N/A')*100:.2f}%" if isinstance(info.get('profitMargins'), (int, float)) else 'N/A',
                        f"{info.get('operatingMargins', 'N/A')*100:.2f}%" if isinstance(info.get('operatingMargins'), (int, float)) else 'N/A',
                        f"{info.get('returnOnEquity', 'N/A')*100:.2f}%" if isinstance(info.get('returnOnEquity'), (int, float)) else 'N/A',
                        f"{info.get('returnOnAssets', 'N/A')*100:.2f}%" if isinstance(info.get('returnOnAssets'), (int, float)) else 'N/A',
                        f"{info.get('ebitdaMargins', 'N/A')*100:.2f}%" if isinstance(info.get('ebitdaMargins'), (int, float)) else 'N/A',
                        f"{info.get('grossMargins', 'N/A')*100:.2f}%" if isinstance(info.get('grossMargins'), (int, float)) else 'N/A',
                        format_large_number(info.get('freeCashflow', 'N/A')),
                        format_large_number(info.get('operatingCashflow', 'N/A'))
                    ]
                }
                st.subheader("Profitability Metrics")
                st.table(pd.DataFrame(profitability_data))
        
        with fin_tab2:
            # Financial growth metrics
            growth_data = {
                "Metric": [
                    "Revenue Growth", "Earnings Growth", 
                    "EPS Growth", "Free Cash Flow Growth", 
                    "EBITDA Growth", "Gross Profit Growth"
                ],
                "YoY Growth": [
                    f"{info.get('revenueGrowth', 'N/A')*100:.2f}%" if isinstance(info.get('revenueGrowth'), (int, float)) else 'N/A',
                    f"{info.get('earningsGrowth', 'N/A')*100:.2f}%" if isinstance(info.get('earningsGrowth'), (int, float)) else 'N/A',
                    f"{info.get('earningsQuarterlyGrowth', 'N/A')*100:.2f}%" if isinstance(info.get('earningsQuarterlyGrowth'), (int, float)) else 'N/A',
                    "N/A",  # FCF growth typically requires calculation
                    "N/A",  # EBITDA growth typically requires calculation
                    "N/A"   # Gross profit growth typically requires calculation
                ]
            }
            st.subheader("Growth Metrics (Year-over-Year)")
            st.table(pd.DataFrame(growth_data))
            
            # If we have revenue and earnings data, create growth charts
            if 'revenueGrowth' in info and 'earningsGrowth' in info:
                st.subheader("Growth Visualization")
                growth_chart_data = {
                    "Metric": ["Revenue", "Earnings", "EPS"],
                    "Growth Rate (%)": [
                        info.get('revenueGrowth', 0) * 100,
                        info.get('earningsGrowth', 0) * 100,
                        info.get('earningsQuarterlyGrowth', 0) * 100
                    ]
                }
                
                growth_df = pd.DataFrame(growth_chart_data)
                
                # Create bar chart for growth rates
                growth_fig = px.bar(
                    growth_df,
                    x="Metric",
                    y="Growth Rate (%)",
                    color="Growth Rate (%)",
                    title="YoY Growth Rates",
                    color_continuous_scale="Viridis"
                )
                
                growth_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Metric",
                    yaxis_title="Growth Rate (%)",
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=300
                )
                
                st.plotly_chart(growth_fig, use_container_width=True)
                    
        with fin_tab3:
            # Balance sheet metrics
            balance_sheet_data = {
                "Metric": [
                    "Total Cash", "Total Debt", 
                    "Debt-to-Equity", "Current Ratio", 
                    "Quick Ratio", "Book Value Per Share", 
                    "Total Assets", "Total Liabilities"
                ],
                "Value": [
                    format_large_number(info.get('totalCash', 'N/A')),
                    format_large_number(info.get('totalDebt', 'N/A')),
                    f"{info.get('debtToEquity', 'N/A')}" if isinstance(info.get('debtToEquity'), (int, float)) else 'N/A',
                    f"{info.get('currentRatio', 'N/A')}" if isinstance(info.get('currentRatio'), (int, float)) else 'N/A',
                    f"{info.get('quickRatio', 'N/A')}" if isinstance(info.get('quickRatio'), (int, float)) else 'N/A',
                    f"${info.get('bookValue', 'N/A')}" if isinstance(info.get('bookValue'), (int, float)) else 'N/A',
                    format_large_number(info.get('totalAssets', 'N/A')),
                    format_large_number(info.get('totalLiabilities', 'N/A'))
                ]
            }
            st.subheader("Balance Sheet Metrics")
            st.table(pd.DataFrame(balance_sheet_data))
        
        # Recommendation section
        if 'recommendationMean' in info:
            st.subheader("Analyst Recommendations")
            
            # Create gauge chart for analyst recommendations
            recommendation = info.get('recommendationMean', 3)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendation,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Recommendation Rating"},
                gauge={
                    'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "rgba(0, 255, 255, 0.7)"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [1, 1.8], 'color': '#2ECC71'},  # Strong Buy
                        {'range': [1.8, 2.6], 'color': '#82E0AA'},  # Buy
                        {'range': [2.6, 3.4], 'color': '#F7DC6F'},  # Hold
                        {'range': [3.4, 4.2], 'color': '#F5B041'},  # Sell
                        {'range': [4.2, 5], 'color': '#E74C3C'}   # Strong Sell
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': recommendation
                    }
                }
            ))
            
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20),
                height=300,
                annotations=[
                    dict(
                        x=0.1,
                        y=0.2,
                        text="Strong Buy",
                        showarrow=False
                    ),
                    dict(
                        x=0.3,
                        y=0.2,
                        text="Buy",
                        showarrow=False
                    ),
                    dict(
                        x=0.5,
                        y=0.2,
                        text="Hold",
                        showarrow=False
                    ),
                    dict(
                        x=0.7,
                        y=0.2,
                        text="Sell",
                        showarrow=False
                    ),
                    dict(
                        x=0.9,
                        y=0.2,
                        text="Strong Sell",
                        showarrow=False
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendation details if available
            if 'recommendationKey' in info:
                st.info(f"Analyst Consensus: **{info['recommendationKey'].upper()}**")
            
            # Display target price if available
            if 'targetMeanPrice' in info and 'currentPrice' in info:
                target = info['targetMeanPrice']
                current = info['currentPrice']
                upside = ((target / current) - 1) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current:.2f}")
                with col2:
                    st.metric("Target Price", f"${target:.2f}")
                with col3:
                    st.metric("Potential Upside", f"{upside:.2f}%")
        
        # Footer with disclaimer
        st.markdown("---")
        st.caption("""
        **Disclaimer**: This tool provides financial information for educational purposes only. It is not intended to provide investment advice.
        Data is sourced from Yahoo Finance and may be delayed. Always conduct thorough research before making investment decisions.
        """)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please check the stock symbol and try again.")
