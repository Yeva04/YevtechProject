# Academic Predictive System Handover Instructions

## Email Configuration
- The app uses Gmail SMTP to send emails (e.g., complaint responses).
- Current settings are stored in `.env`:

- **Action Required**:
1. Create a project-specific Gmail account (e.g., academicapp25@gmail.com).
2. Enable 2-Step Verification (`https://myaccount.google.com/security`).
3. Generate an App Password for "Mail" (Custom name: "SAPA app").
4. Update `.env` with the new credentials:

## Running the App
- Install dependencies: `pip install -r requirements.txt`
- Run: `python app.py`
- Access: `http://192.168.82.129:5000`


## Email Configuration
- Emails are sent via Gmail SMTP (port 465, SSL).
- Credentials are stored in `.env`:
