from flask import make_response

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "https://kdmotoshop.onrender.com")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "https://kdmotoshop.onrender.com")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response