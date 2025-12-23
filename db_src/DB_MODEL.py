from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Represents a user in the system with authentication details."""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    fname = db.Column(db.String(70))
    lname = db.Column(db.String(70))
    type = db.Column(db.String(20))

class X_ray(UserMixin, db.Model):
    """Represents an X-ray image record associated with a patient and a user."""
    img_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    Patient_Name = db.Column(db.String(200))
    image_adder = db.Column(db.String(300))

def create_user(fname, lname, email, password, type):
    """Creates a new user record with a hashed password and saves it to the database."""
    new_user = User(fname=fname, lname=lname, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'), type=type)
    db.session.add(new_user)
    db.session.commit()

def authentication_user(email, password):
    """Authenticates a user by verifying the email exists and the password matches."""
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return False
    return True

def get_usertype(email):
    """Retrieves the type/role of a user based on their email address."""
    user = User.query.filter_by(email=email).first()
    return user.type

def User_data(email):
    """Retrieves the full User object corresponding to the given email."""
    user = User.query.filter_by(email=email).first()
    return user

def save_image(user_id, Patient_Name, image_adder):
    """Saves a new X-ray record to the database and returns success status."""
    new_image = X_ray(user_id=user_id, Patient_Name=Patient_Name, image_adder=image_adder)
    db.session.add(new_image)
    db.session.commit()
    return 'ok'

def get_images(u_id):
    """Retrieves a list of patient names for X-rays uploaded by a specific user."""
    uploads = X_ray.query.filter_by(user_id=u_id).all()
    names = db.session.query(X_ray.Patient_Name).filter(X_ray.user_id == u_id).all()
    for x in range(len(names)):
        y = str(names[x])
        names[x] = y[2:len(y)-3:]
    if not uploads:
        return None
    else:
        return names

def get_users(u_id):
    """Retrieves a list of emails of all users except the user with the specified ID."""
    user_listed = db.session.query(User.email).filter(User.id != u_id).all()
    for x in range(len(user_listed)):
        y = str(user_listed[x])
        user_listed[x] = y[2:len(y)-3:]
    if not user_listed:
        return None
    else:
        return user_listed

def delete_user(u_email):
    """Deletes a user and their associated X-ray records from the database using their email."""
    record_obj = db.session.query(User).filter(User.email == u_email).first()
    db.session.delete(record_obj)
    db.session.commit()
    images_delete = db.session.query(X_ray).filter(X_ray.user_id == record_obj.id).all()
    if images_delete:
        for img in images_delete:
            db.session.delete(img)
        db.session.commit()

def getpath(p_name):
    """Extracts the filename from the stored path for a given patient's X-ray."""
    image_addr = db.session.query(X_ray.image_adder).filter(X_ray.Patient_Name == p_name).first()
    if not image_addr:
        return ""
    return image_addr[0].split("/")[-1]

def get_username(user_id):
    """Retrieves and formats the full name of a user given their user ID."""
    fname = db.session.query(User.fname).filter(User.id == user_id).first()
    lname = db.session.query(User.lname).filter(User.id == user_id).first()
    u_name = str(fname[0]) + ' ' + str(lname[0])
    return u_name