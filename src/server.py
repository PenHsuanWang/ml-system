# src/server.py

from fastapi import FastAPI

from webapp import data_io_serving_app_router
from webapp import ml_training_serving_app_router


app = FastAPI()

# Include the router in the main FastAPI app
app.include_router(data_io_serving_app_router.router)
app.include_router(ml_training_serving_app_router.router)

