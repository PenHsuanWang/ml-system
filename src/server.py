# src/server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp import data_io_serving_app_router
from webapp import ml_training_serving_app_router
from webapp import stock_price_analyzer_serving_app_route
from webapp import mlflow_model_download_serving_app_route


origins = [
    "http://localhost:3000",  # for react application development
]

app = FastAPI()

# Include the router in the main FastAPI app
app.include_router(data_io_serving_app_router.router)
app.include_router(ml_training_serving_app_router.router)
app.include_router(stock_price_analyzer_serving_app_route.router)
app.include_router(mlflow_model_download_serving_app_route.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

