from fastapi import APIRouter
from fastapi import Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.data_io.data_sinker.sinker import DataSinkerFactory

class InitDataFetcherRequest(BaseModel):
    data_fetcher_name: str
    data_source_type: str


class InitDataSinkerRequest(BaseModel):
    data_sinker_name: str
    data_sink_type: str


class FetchDataFromSourceRequest(BaseModel):
    data_fetcher_name: str
    args: list
    kwargs: dict


class GetAsDataframeRequest(BaseModel):
    data_fetcher_name: str
    args: list
    kwargs: dict


class DataIOServingApp:
    router = APIRouter()

    def __init__(self):
        self._data_fetcher = {}
        self._data_sinker = {}

    # define singleton pattern to get the app instance
    @staticmethod
    def get_app():
        if not hasattr(DataIOServingApp, "_app"):
            DataIOServingApp._app = DataIOServingApp()
        return DataIOServingApp._app

# Define get_app dependency
def get_app():
    return DataIOServingApp.get_app()


# API Router functions
@DataIOServingApp.router.post("/dataio_manager/init_data_fetcher")
def init_data_fetcher(requests: InitDataFetcherRequest = Body(...), app: DataIOServingApp = Depends(get_app)):
    """
    create a data fetcher instance
    provide the data source type and data source dir
    :param requests: pydantic model to parse the request
    :return:
    """
    data_fetcher_name = requests.data_fetcher_name
    data_source_type = requests.data_source_type

    # check the fetcher name is not duplicated, return http error if duplicated
    if data_fetcher_name in app._data_fetcher:
        return JSONResponse(status_code=400, content={"message": f"Data Fetcher named: {data_fetcher_name} already exists"})

    fetcher = DataFetcherFactory.create_data_fetcher(data_source_type)
    app._data_fetcher[data_fetcher_name] = fetcher

    return {"message": f"Data Fetcher named: {data_fetcher_name} created successfully, added into fetcher list"}


@DataIOServingApp.router.get("/dataio_manager/init_data_sinker")
def init_data_sinker(requests: InitDataSinkerRequest = Body(...), app: DataIOServingApp = Depends(get_app)):
    """
    create a data sinker instance
    provide the data sink type and data sink dir
    :param requests: pydantic model to parse the request
    :return:
    """
    data_sinker_name = requests.data_sinker_name
    data_sink_type = requests.data_sink_type

    # check the sinker name is not duplicated, return http error if duplicated
    if data_sinker_name in app._data_sinker:
        return JSONResponse(status_code=400, content={"message": f"Data Sinker named: {data_sinker_name} already exists"})

    sinker = DataSinkerFactory.create_data_sinker(data_sink_type)
    app._data_sinker[data_sinker_name] = sinker

    return {"message": f"Data Sinker named: {data_sinker_name} created successfully, added into sinker list"}


# API Router function to get the data fetcher list and data sinker list
@DataIOServingApp.router.get("/dataio_manager/get_data_fetcher_list")
def get_data_fetcher_list(app: DataIOServingApp = Depends(get_app)):
    """
    get the data fetcher list
    :param app:
    :return:
    """
    return {"data_fetcher_list": list(app._data_fetcher.keys())}


@DataIOServingApp.router.get("/dataio_manager/get_data_sinker_list")
def get_data_sinker_list(app: DataIOServingApp = Depends(get_app)):
    """
    get the data sinker list
    :return:
    """
    return {"data_sinker_list": list(app._data_sinker.keys())}


# Implement the API Router function for data fetcher
@DataIOServingApp.router.post("/dataio_manager/fetch_data_from_source")
def fetch_data_from_source(requests: FetchDataFromSourceRequest = Body(...), app: DataIOServingApp = Depends(get_app)):
    """
    fetch data from data source
    :param requests: pydantic model to parse the request
    :return:
    """

    data_fetcher_name = requests.data_fetcher_name
    args = requests.args
    kwargs = requests.kwargs

    if data_fetcher_name not in app._data_fetcher:
        return JSONResponse(
            status_code=400,
            content={"message": f"Data Fetcher named: {data_fetcher_name} not found"}
        )

    app._data_fetcher[data_fetcher_name].fetch_from_source(*args, **kwargs)

    return {"message": f"Data Fetcher named: {data_fetcher_name} fetch data successfully"}

@DataIOServingApp.router.post("/dataio_manager/get_as_dataframe")
def get_as_dataframe(requests: GetAsDataframeRequest = Body(...), app: DataIOServingApp = Depends(get_app)):
    """
    get data as dataframe
    :param requests: pydantic model to parse the request
    :return:
    """

    data_fetcher_name = requests.data_fetcher_name
    args = requests.args
    kwargs = requests.kwargs

    if data_fetcher_name not in app._data_fetcher:
        return JSONResponse(
            status_code=400,
            content={"message": f"Data Fetcher named: {data_fetcher_name} not found"}
        )

    pd_get_from_fetcher = app._data_fetcher[data_fetcher_name].get_as_dataframe(**kwargs)

    # return dataframes as json match rest api response
    return pd_get_from_fetcher.to_json(orient="records")

# Implement the API Router function for data sinker
# TODO: implement the data sinker API Router function



