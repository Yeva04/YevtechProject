from flask import Flask
from flask_mail import Mail, Message
import logging
from dotenv import load_dotenv
import os

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Gmail SMTP configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

def test_email():
    with app.app_context():
        try:
            msg = Message(
                subject='Test Email from Flask',
                recipients=['kayloojo6@gmail.com'],  # Replace with a test email you can access
                body='This is a test email sent from the Flask app.'
            )
            mail.send(msg)
            logger.info("Test email sent successfully.")
            print("Test email sent successfully!")
        except Exception as e:
            logger.error(f"Error sending test email: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_email()