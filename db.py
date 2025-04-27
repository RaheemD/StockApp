import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

# Database connection setup - using SQLite for local storage
# This will work without requiring an external PostgreSQL server
DATABASE_URL = "sqlite:///stock_analysis.db"
engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
class Stock(Base):
    """Stock symbols and basic info"""
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    company_name = Column(String(100))
    sector = Column(String(50))
    industry = Column(String(100))
    country = Column(String(50))
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    user_favorites = relationship("UserFavorite", back_populates="stock", cascade="all, delete-orphan")
    saved_analyses = relationship("SavedAnalysis", back_populates="stock", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', company_name='{self.company_name}')>"


class StockPrice(Base):
    """Historical stock price data"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adjusted_close = Column(Float)
    volume = Column(Float)
    
    # Relationship
    stock = relationship("Stock", back_populates="prices")
    
    def __repr__(self):
        return f"<StockPrice(symbol='{self.stock.symbol}', date='{self.date}', close='{self.close}')>"


class UserFavorite(Base):
    """User favorite stocks for quick access"""
    __tablename__ = 'user_favorites'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)  # Could be linked to a user table in the future
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date_added = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationship
    stock = relationship("Stock", back_populates="user_favorites")
    
    def __repr__(self):
        return f"<UserFavorite(user='{self.user_id}', stock='{self.stock.symbol}')>"


class SavedAnalysis(Base):
    """Saved analysis configurations and results"""
    __tablename__ = 'saved_analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)  # Could be linked to a user table in the future
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    analysis_name = Column(String(100))
    date_created = Column(DateTime, default=datetime.utcnow)
    period = Column(String(20))  # e.g., '1mo', '3mo', '1y', etc.
    interval = Column(String(10))  # e.g., '1d', '1h', etc.
    indicators = Column(Text)  # Stored as JSON
    chart_type = Column(String(20))
    notes = Column(Text)
    
    # Relationship
    stock = relationship("Stock", back_populates="saved_analyses")
    
    def get_indicators(self):
        """Convert the JSON stored indicators back to a list"""
        if self.indicators:
            return json.loads(self.indicators)
        return []
    
    def set_indicators(self, indicators_list):
        """Convert a list of indicators to JSON for storage"""
        if indicators_list:
            self.indicators = json.dumps(indicators_list)
    
    def __repr__(self):
        return f"<SavedAnalysis(name='{self.analysis_name}', stock='{self.stock.symbol}')>"


# Functions to interact with the database
def init_db():
    """Initialize the database by creating all tables"""
    Base.metadata.create_all(engine)


def add_or_update_stock(symbol, company_info):
    """Add a new stock to the database or update if it exists"""
    session = Session()
    
    try:
        # Check if stock already exists
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        
        if not stock:
            # Create new stock
            stock = Stock(
                symbol=symbol,
                company_name=company_info.get('longName', company_info.get('shortName', symbol)),
                sector=company_info.get('sector'),
                industry=company_info.get('industry'),
                country=company_info.get('country'),
                last_updated=datetime.utcnow()
            )
            session.add(stock)
        else:
            # Update existing stock
            stock.company_name = company_info.get('longName', company_info.get('shortName', symbol))
            stock.sector = company_info.get('sector')
            stock.industry = company_info.get('industry')
            stock.country = company_info.get('country')
            stock.last_updated = datetime.utcnow()
        
        session.commit()
        return stock.id
    except Exception as e:
        session.rollback()
        print(f"Error adding/updating stock: {e}")
        return None
    finally:
        session.close()


def add_stock_prices(stock_id, price_data):
    """Add historical price data for a stock"""
    session = Session()
    try:
        # Convert the data to records
        data_records = []
        
        # Convert to simple Python types first
        dates = price_data.index.to_pydatetime()
        opens = price_data['Open'].values
        highs = price_data['High'].values
        lows = price_data['Low'].values
        closes = price_data['Close'].values
        adj_closes = price_data['Adj Close'].values
        volumes = price_data['Volume'].values
        
        # Create records
        for i in range(len(dates)):
            try:
                record = {
                    'stock_id': stock_id,
                    'date': dates[i],
                    'open': float(opens[i]) if not pd.isna(opens[i]) else None,
                    'high': float(highs[i]) if not pd.isna(highs[i]) else None,
                    'low': float(lows[i]) if not pd.isna(lows[i]) else None,
                    'close': float(closes[i]) if not pd.isna(closes[i]) else None,
                    'adjusted_close': float(adj_closes[i]) if not pd.isna(adj_closes[i]) else None,
                    'volume': float(volumes[i]) if not pd.isna(volumes[i]) else None
                }
                data_records.append(record)
            except Exception as e:
                print(f"Error processing record for date {dates[i]}: {e}")
                continue
        
        # Add or update records
        for record in data_records:
            try:
                existing_price = session.query(StockPrice).filter(
                    StockPrice.stock_id == stock_id,
                    StockPrice.date == record['date']
                ).first()
                
                if existing_price:
                    # Update existing record
                    for key, value in record.items():
                        setattr(existing_price, key, value)
                else:
                    # Create new record
                    new_price = StockPrice(**record)
                    session.add(new_price)
            except Exception as e:
                print(f"Error saving record for date {record['date']}: {e}")
                continue
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding stock prices: {e}")
        return False
    finally:
        session.close()


def get_stock_by_symbol(symbol):
    """Get a stock by its symbol"""
    session = Session()
    try:
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        return stock
    except Exception as e:
        print(f"Error getting stock: {e}")
        return None
    finally:
        session.close()


def add_to_favorites(user_id, symbol, notes=None):
    """Add a stock to user favorites"""
    session = Session()
    try:
        # Get stock by symbol
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            return False
        
        # Check if already a favorite
        existing = session.query(UserFavorite).filter(
            UserFavorite.user_id == user_id,
            UserFavorite.stock_id == stock.id
        ).first()
        
        if not existing:
            favorite = UserFavorite(
                user_id=user_id,
                stock_id=stock.id,
                notes=notes
            )
            session.add(favorite)
            session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding to favorites: {e}")
        return False
    finally:
        session.close()


def remove_from_favorites(user_id, symbol):
    """Remove a stock from user favorites"""
    session = Session()
    try:
        # Get stock by symbol
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            return False
        
        # Find and delete the favorite
        favorite = session.query(UserFavorite).filter(
            UserFavorite.user_id == user_id,
            UserFavorite.stock_id == stock.id
        ).first()
        
        if favorite:
            session.delete(favorite)
            session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error removing from favorites: {e}")
        return False
    finally:
        session.close()


def get_user_favorites(user_id):
    """Get all favorite stocks for a user"""
    session = Session()
    try:
        favorites = session.query(UserFavorite).filter(
            UserFavorite.user_id == user_id
        ).all()
        
        # Return a list of stock symbols and names
        result = []
        for fav in favorites:
            stock = session.query(Stock).filter(Stock.id == fav.stock_id).first()
            if stock:
                result.append({
                    "symbol": stock.symbol,
                    "company_name": stock.company_name,
                    "notes": fav.notes,
                    "date_added": fav.date_added
                })
        return result
    except Exception as e:
        print(f"Error getting favorites: {e}")
        return []
    finally:
        session.close()


def save_analysis(user_id, stock_symbol, analysis_name, period, interval, indicators, chart_type, notes=None):
    """Save an analysis configuration"""
    session = Session()
    try:
        # Get stock by symbol
        stock = session.query(Stock).filter(Stock.symbol == stock_symbol).first()
        if not stock:
            return False
        
        # Create new saved analysis
        analysis = SavedAnalysis(
            user_id=user_id,
            stock_id=stock.id,
            analysis_name=analysis_name,
            period=period,
            interval=interval,
            chart_type=chart_type,
            notes=notes
        )
        
        # Set indicators
        analysis.set_indicators(indicators)
        
        session.add(analysis)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error saving analysis: {e}")
        return False
    finally:
        session.close()


def get_saved_analyses(user_id):
    """Get all saved analyses for a user"""
    session = Session()
    try:
        analyses = session.query(SavedAnalysis).filter(
            SavedAnalysis.user_id == user_id
        ).all()
        
        # Return detailed info
        result = []
        for analysis in analyses:
            stock = session.query(Stock).filter(Stock.id == analysis.stock_id).first()
            if stock:
                result.append({
                    "id": analysis.id,
                    "name": analysis.analysis_name,
                    "symbol": stock.symbol,
                    "company_name": stock.company_name,
                    "period": analysis.period,
                    "interval": analysis.interval,
                    "indicators": analysis.get_indicators(),
                    "chart_type": analysis.chart_type,
                    "notes": analysis.notes,
                    "date_created": analysis.date_created
                })
        return result
    except Exception as e:
        print(f"Error getting saved analyses: {e}")
        return []
    finally:
        session.close()


def get_cached_price_data(symbol, period='1mo'):
    """Get cached price data for a stock if available"""
    session = Session()
    try:
        # Find the stock
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            return None
        
        # Get price data
        prices = session.query(StockPrice).filter(
            StockPrice.stock_id == stock.id
        ).order_by(StockPrice.date).all()
        
        if not prices:
            return None
        
        # Convert to DataFrame
        data = {
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Adj Close': [],
            'Volume': []
        }
        index = []
        
        for price in prices:
            index.append(price.date)
            data['Open'].append(price.open)
            data['High'].append(price.high)
            data['Low'].append(price.low)
            data['Close'].append(price.close)
            data['Adj Close'].append(price.adjusted_close)
            data['Volume'].append(price.volume)
        
        df = pd.DataFrame(data, index=index)
        return df
    except Exception as e:
        print(f"Error getting cached price data: {e}")
        return None
    finally:
        session.close()


# Initialize database on import
if __name__ == "__main__":
    init_db()