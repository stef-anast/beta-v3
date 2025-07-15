from sqlalchemy import Column, Integer, String, Float, Date, DateTime, LargeBinary, Boolean
from sqlalchemy.sql import func
from .database import Base

class Match(Base):
    """
    SQLAlchemy model representing a match in the database.
    """
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    match_date = Column(Date, index=True)
    competition = Column(String)
    time = Column(String)
    match_code = Column(String)
    team_home = Column(String, index=True)
    team_away = Column(String, index=True)
    odds_1 = Column(Float)
    odds_x = Column(Float)
    odds_2 = Column(Float)
    score = Column(String, nullable=True)
    result = Column(String, nullable=True)

class TrainedModel(Base):
    """
    SQLAlchemy model for storing trained model binaries in the database.
    """
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    model_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True, nullable=False) 