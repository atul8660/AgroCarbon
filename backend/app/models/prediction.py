from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    species = Column(String)
    health = Column(String)
    
    biomass_kg = Column(Float)
    carbon_kg = Column(Float)
    co2_kg = Column(Float)
    estimated_carbon_credit = Column(Float)
    
    # AWS S3 Storage CDN Pointers
    full_tree_url = Column(String, nullable=True)
    closeup_url = Column(String, nullable=True)
    pdf_report_url = Column(String, nullable=True)
    csv_report_url = Column(String, nullable=True)
    
    prediction_date = Column(DateTime, server_default=func.now())
    
    # Back Propagation Relationships
    user = relationship("User")
    reports = relationship("Report", back_populates="prediction")
