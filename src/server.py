# src/server.py

from fastapi import FastAPI
from webapp.app import Example
from webapp.data_io_serving_app import DataIOServingApp
# Import your Example class from the app module

app = FastAPI()

# Create an instance of the Example class
example = Example()
data_io_serving_app = DataIOServingApp()

# Include the router in the main FastAPI app
app.include_router(example.router)
app.include_router(data_io_serving_app.router)


