from flask_sqlalchemy import SQLAlchemy

from flask_login import UserMixin,current_user
from werkzeug.security import generate_password_hash, check_password_hash
db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    fname = db.Column(db.String(70))
    lname = db.Column(db.String(70))
    type = db.Column(db.String(20))



class X_ray(UserMixin, db.Model):
    img_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    Patient_Name = db.Column(db.String(200))
    image_adder = db.Column(db.String(300))




def create_user(fname,lname, email,password, type):
    new_user = User(fname=fname, lname=lname, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'), type=type)
    db.session.add(new_user)
    db.session.commit()


def authentication_user(email, password):
    user = User.query.filter_by(email=email).first()
    print(user.type)
    if not user or not check_password_hash(user.password,password):
        return False
    return True

def get_usertype(email):
    user = User.query.filter_by(email=email).first()
    return user.type


def User_data(email):
    user = User.query.filter_by(email=email).first()
    return user

def save_image(user_id,Patient_Name,image_adder):
    new_image = X_ray(user_id=user_id, Patient_Name=Patient_Name, image_adder=image_adder)
    db.session.add(new_image)
    db.session.commit()
    return 'ok'


def get_images(u_id):
    uploads = X_ray.query.filter_by(user_id=u_id).all()
    names = db.session.query(X_ray.Patient_Name).filter(X_ray.user_id == u_id).all()
    len(names)
    for x in range(len(names)):
        y = str(names[x])
        l_x = len(y)
        l_x = l_x-3
        names[x] = y[2:l_x:]
    if not uploads:
        return None
    else:
        return names

def get_users(u_id):
    user_listed = db.session.query(User.email).filter(User.id != u_id).all()
    len(user_listed)
    for x in range(len(user_listed)):
        y = str(user_listed[x])
        l_x = len(y)
        l_x = l_x-3
        user_listed[x] = y[2:l_x:]
    if not user_listed:
        return None
    else:
        return user_listed

def delete_user(u_email):
    record_obj = db.session.query(User).filter(User.email == u_email).first()
    print(record_obj)
    db.session.delete(record_obj)
    db.session.commit()
    images_delete = db.session.delete(X_ray).filter(X_ray.user_id == record_obj).all()
    if images_delete:
        db.session.delete(images_delete)
        db.session.commit()

def getpath(p_name):
    image_addr = db.session.query(X_ray.image_adder).filter(X_ray.Patient_Name == p_name).first()
    if len(image_addr) == 0:
        return ""
    return image_addr[0].split("/")[-1]


def get_username(user_id):
    fname = db.session.query(User.fname).filter(User.id == user_id).first()
    lname = db.session.query(User.lname).filter(User.id == user_id).first()
    u_name = str(fname[0])+' '+str(lname[0])
    print (u_name)
    return u_name