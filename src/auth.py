import requests
import os
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

def get_api_key():
    key = os.getenv("FIREBASE_API_KEY")
    if not key:
        return None
    return key.strip()

def sign_up_with_email_and_password(email, password):
    """
    Sign up a new user using Firebase REST API.
    """
    api_key = get_api_key()
    if not api_key:
        return {"error": {"message": "FIREBASE_API_KEY is missing from environment variables."}}
        
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        return {"error": {"message": str(e)}}

def sign_in_with_email_and_password(email, password):
    """
    Sign in an existing user using Firebase REST API.
    """
    api_key = get_api_key()
    if not api_key:
        return {"error": {"message": "FIREBASE_API_KEY is missing from environment variables."}}

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        return {"error": {"message": str(e)}}

def get_user_info(id_token):
    """
    Get user info from Firebase REST API using ID Token.
    """
    api_key = get_api_key()
    if not api_key:
        return {"error": {"message": "FIREBASE_API_KEY is missing from environment variables."}}

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}"
    payload = {
        "idToken": id_token
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        return {"error": {"message": str(e)}}
