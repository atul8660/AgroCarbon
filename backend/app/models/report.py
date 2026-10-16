from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    
    pdf_url = Column(String, nullable=False)
    csv_url = Column(String, nullable=False)
    
    created_at = Column(DateTime, server_default=func.now())
    
    prediction = relationship("Prediction", back_populates="reports")
