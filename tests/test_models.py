import pytest
from db_src.DB_MODEL import User, create_user, authentication_user
from werkzeug.security import check_password_hash

def test_new_user(init_database):
    """
    GIVEN a User model
    WHEN a new User is created
    THEN check the email, hashed password, and user type
    """
    create_user('John', 'Doe', 'test@test.com', 'password', 'Doctor')
    user = User.query.filter_by(email='test@test.com').first()
    
    assert user.email == 'test@test.com'
    assert user.fname == 'John'
    assert user.lname == 'Doe'
    assert user.type == 'Doctor'
    assert check_password_hash(user.password, 'password')

def test_authentication(init_database):
    """
    GIVEN a registered user
    WHEN authentication_user is called
    THEN check if it returns True for correct password and False for incorrect
    """
    create_user('Jane', 'Doe', 'jane@test.com', 'securepass', 'Doctor')
    
    assert authentication_user('jane@test.com', 'securepass') == True
    assert authentication_user('jane@test.com', 'wrongpass') == False
    assert authentication_user('nonexistent@test.com', 'securepass') == False