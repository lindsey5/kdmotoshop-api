from datetime import datetime
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from services.predict_service import forecast_items_qty_sold, predict_future_sales

predict_router = APIRouter()

@predict_router.get("/api/predict")
async def predict(
    month: int = Query(default=(datetime.now().month % 12) + 1),
    year: int = Query(default=datetime.now().year + (1 if datetime.now().month == 12 else 0))
):
    try:
        forecast_data = predict_future_sales(month, year)
        return JSONResponse(content=forecast_data)
    except Exception as e:
        print("Error", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

@predict_router.get("/api/predict/items")
async def predict_items(
    month: int = Query(default=(datetime.now().month % 12) + 1),
    year: int = Query(default=datetime.now().year + (1 if datetime.now().month == 12 else 0))
):
    try:
        forecast_data = forecast_items_qty_sold(month, year)
        return JSONResponse(content=forecast_data)
    except Exception as e:
        print("Error", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
