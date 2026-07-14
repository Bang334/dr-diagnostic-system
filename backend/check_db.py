import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure console output uses UTF-8
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.core.database import SessionLocal
from app.models.patient import Patient

db = SessionLocal()
try:
    patients = db.query(Patient).all()
    print(f"Total patients in DB: {len(patients)}")
    for p in patients:
        print(f"- ID: {p.id} | Code: {p.patient_code} | Name: {p.full_name} | Phone: {p.phone_number} | DOB: {p.date_of_birth}")
except Exception as e:
    print(f"Error querying database: {e}")
finally:
    db.close()
