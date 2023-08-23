from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import json

from src.webapp.stock_price_analyzer_serving_app import StockAnalyzerServingApp, SafeEncoder

router = APIRouter()


# Pydantic models for request bodies
class InitStockAnalyzerBody(BaseModel):
    stock_id: str
    start_date: str
    end_date: str


class MovingAverageBody(BaseModel):
    window_size: int


# Dependency to get or create app instance
def get_app(stock_id: str = None, start_date: str = None, end_date: str = None):
    if not hasattr(get_app, "_instance"):
        if stock_id and start_date and end_date:
            get_app._instance = StockAnalyzerServingApp(stock_id, start_date, end_date)
        else:
            raise HTTPException(status_code=400, detail="StockAnalyzerServingApp not initialized")
    return get_app._instance


# REST API endpoints
@router.post("/stock_analyzer/init")
def init_stock_analyzer(request: InitStockAnalyzerBody):
    app = get_app(request.stock_id, request.start_date, request.end_date)
    return {"message": "StockAnalyzerServingApp initialized successfully"}


@router.post("/stock_analyzer/calculate_moving_average")
def calculate_moving_average(request: MovingAverageBody, stock_analyzer_app: StockAnalyzerServingApp = Depends(get_app)):
    stock_analyzer_app.calculate_moving_average(request.window_size)
    return {"message": f"Moving average for window size {request.window_size} calculated successfully"}


@router.get("/stock_analyzer/calculate_daily_return_percentage")
def calculate_daily_return_percentage(stock_analyzer_app: StockAnalyzerServingApp = Depends(get_app)):
    stock_analyzer_app.calculate_daily_return_percentage()
    return {"message": "Daily return percentage calculated successfully"}


@router.get("/stock_analyzer/get_analysis_data")
def get_analysis_data(stock_analyzer_app: StockAnalyzerServingApp = Depends(get_app)):
    encoded_json_data = stock_analyzer_app.get_encoded_str_analysis_data()
    if encoded_json_data is not None:
        return Response(content=encoded_json_data, media_type="application/json")
    else:
        raise HTTPException(status_code=400, detail="No analysis data available")

