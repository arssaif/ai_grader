import pytest

def test_home_page_redirect(client):
    """
    GIVEN a Flask application
    WHEN the '/' page is requested (GET)
    THEN check that it redirects to login (since user is not logged in)
    """
    response = client.get('/', follow_redirects=True)
    assert response.status_code == 200
    assert b"Login" in response.data

def test_login_page(client):
    """
    GIVEN a Flask application
    WHEN the '/login' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get('/login')
    assert response.status_code == 200
    assert b"Login" in response.data

def test_signup_page(client):
    """
    GIVEN a Flask application
    WHEN the '/signup' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get('/signup')
    assert response.status_code == 200
    assert b"SIGN-UP NOW" in response.data