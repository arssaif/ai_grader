from flask_restful import Resource
from db_src.DB_MODEL import create_user, authentication_user, User_data, \
                                                save_image, get_usertype
from flask import jsonify, request
from flask_login import login_user, current_user
import os

APP_ROOT = os.path.abspath(os.curdir)

class RegisterUser(Resource):
    """
    API resource for registering a new user.
    """
    def post(self):
        """
        Handles POST requests for user registration.
        Expects form data: fname, lname, email, password.
        """
        print(request.form)
        print(request.data)
        fname = request.form.get("fname")
        lname = request.form.get("lname")
        email = request.form.get("email")
        password = request.form.get("password")
        create_user(fname, lname, email, password, "patient")
        return "User registered"


class LoginUser(Resource):
    """
    API resource for authenticating and logging in a user.
    """
    def post(self):
        """
        Handles POST requests for user login.
        Expects form data: email, password.
        """
        if request.form:
            email = request.form.get('email')
            password = request.form.get("password")
            resp = authentication_user(email, password)
            print(resp)
            if resp:
                User_Type = get_usertype(email)
                u_data = User_data(email)
                login_user(u_data)
                return User_Type
            else:
                return "not ok"