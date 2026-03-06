# main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import datetime

# --- Database Configuration ---
# Remember to replace 'your_strong_password' with your actual password
DATABASE_URL = "postgresql://feedback_user:Fds1584@localhost/feedback_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- API Initialization ---
app = FastAPI(title="User Feedback & Review Module")
templates = Jinja2Templates(directory="templates")

# --- Database Table Model (SQLAlchemy) ---
class ReportedSMS(Base):
    __tablename__ = "reported_sms"
    id = Column(Integer, primary_key=True, index=True)
    message_content = Column(Text, nullable=False)
    reported_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    status = Column(String(50), default="pending_review")

# --- API Data Model (Pydantic) ---
class ReportRequest(BaseModel):
    message: str

# Create the table in the database if it doesn't exist
Base.metadata.create_all(bind=engine)

# --- API Endpoints ---

# Endpoint for the public-facing submission form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page for submitting feedback."""
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint to handle new report submissions
@app.post("/report")
async def create_report(request: ReportRequest):
    """Receives a new SMS report and saves it to the database."""
    db = SessionLocal()
    try:
        new_report = ReportedSMS(message_content=request.message)
        db.add(new_report)
        db.commit()
        return {"message": "Thank you! Your report has been submitted successfully."}
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})
    finally:
        db.close()

# --- NEW ENDPOINTS FOR REVIEW INTERFACE ---

@app.get("/review", response_class=HTMLResponse)
async def review_reports(request: Request):
    """Serves the HTML page for reviewing pending reports."""
    db = SessionLocal()
    try:
        # Fetch all messages with the status 'pending_review'
        pending_reports = db.query(ReportedSMS).filter(ReportedSMS.status == 'pending_review').order_by(ReportedSMS.reported_at.desc()).all()
        return templates.TemplateResponse("review.html", {"request": request, "reports": pending_reports})
    finally:
        db.close()

@app.post("/approve/{report_id}")
async def approve_report(report_id: int):
    """Updates the status of a report to 'approved'."""
    db = SessionLocal()
    try:
        report = db.query(ReportedSMS).filter(ReportedSMS.id == report_id).first()
        if report:
            report.status = 'approved'
            db.commit()
            return {"message": f"Report {report_id} approved."}
        return JSONResponse(status_code=404, content={"detail": "Report not found."})
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})
    finally:
        db.close()

@app.post("/reject/{report_id}")
async def reject_report(report_id: int):
    """Updates the status of a report to 'rejected'."""
    db = SessionLocal()
    try:
        report = db.query(ReportedSMS).filter(ReportedSMS.id == report_id).first()
        if report:
            report.status = 'rejected'
            db.commit()
            return {"message": f"Report {report_id} rejected."}
        return JSONResponse(status_code=404, content={"detail": "Report not found."})
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})
    finally:
        db.close()
