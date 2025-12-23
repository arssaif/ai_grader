"""
Configuration module for the Flask application.

This module initializes the Flask app, sets up the database configuration using SQLAlchemy,
configures the RESTful API, and initializes the LoginManager for user authentication.
"""

import os
from db_src.DB_MODEL import (
    db, User
)
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

from flask_login import LoginManager, login_user, logout_user, login_required, current_user

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(APP_ROOT, 'templates'), static_folder=os.path.join(APP_ROOT, 'static'))
app.secret_key = "mmmz1234"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_POOL_RECYCLE'] = 499
app.config['SQLALCHEMY_POOL_TIMEOUT'] = None
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

api = Api(app)

db.init_app(app)

with app.app_context():
    db.create_all()

app.app_context().push()

# Login Manager Configuration
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(email):
    """
    Reload the user object from the user ID stored in the session.

    Args:
        email (str): The unique identifier (email) for the user.

    Returns:
        User: The User object associated with the provided email, or None if not found.
    """
    user = User.query.get(email)
    return user