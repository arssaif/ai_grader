import os
from db_src.DB_MODEL import (
    db, User
)
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

from flask_login import LoginManager,login_user,logout_user,login_required,current_user
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key="mmmz1234"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
app.config['SQLALCHEMY_POOL_RECYCLE']=499
app.config['SQLALCHEMY_POOL_TIMEOUT']=None
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


api = Api(app)

db.init_app(app)
#db = SQLAlchemy(app=app)
#db = SQLAlchemy(app)
#db.create_all()
with app.app_context():
    db.create_all()

app.app_context().push()


#with app.app_context().push()
login_manager = LoginManager()
login_manager.login_view='login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(email):
    user = User.query.get(email)
    return user