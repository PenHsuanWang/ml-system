from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

from src.webapp.stock_price_analyzer_serving_app import StockAnalyzerServingApp

router = APIRouter()


# Pydantic models for request bodies
class FullAnalysisBody(BaseModel):
    stock_id: str
    start_date: str
    end_date: str
    window_sizes: List[int]


@router.post("/stock_analyzer/full_analysis")
def full_analysis(request: FullAnalysisBody):
    """
    The full analysis endpoint for stock analyzer.
    :param request:
    :return:
    """
    # Step 1: Initialize StockAnalyzerServingApp
    app = StockAnalyzerServingApp(request.stock_id, request.start_date, request.end_date)

    # Step 2: Calculate moving averages for multiple window sizes
    for window_size in request.window_sizes:
        app.calculate_moving_average(window_size)

    # Step 3: Calculate daily return percentage
    app.calculate_daily_return_percentage()

    # Step 4: Get analysis data
    encoded_json_data = app.get_encoded_str_analysis_data()
    if encoded_json_data is not None:
        return Response(content=encoded_json_data, media_type="application/json")
    else:
        raise HTTPException(status_code=400, detail="No analysis data available")


