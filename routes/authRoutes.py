from config import api
from views.AuthViews import (
    RegisterUser, LoginUser
)
api.add_resource(RegisterUser, "/api/signup")
api.add_resource(LoginUser, "/api/login")