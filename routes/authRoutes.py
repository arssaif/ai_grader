"""
This module defines the authentication routes for the API.
It maps the authentication views to their respective endpoints.
"""
from config import api
from views.AuthViews import (
    RegisterUser, LoginUser
)

api.add_resource(RegisterUser, "/api/signup")
api.add_resource(LoginUser, "/api/login")