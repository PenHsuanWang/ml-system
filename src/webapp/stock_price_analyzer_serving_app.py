import json
import pandas as pd

from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.data_analyzer.stock_analyzer import StockPriceAnalyzer


class StockAnalyzerServingApp:
    """
    The Application for serving the stock analyzer. Using the internal module StockPriceAnalyzer to do logic process.
    Compose the data analysis end-to-end process. Integrate with the data io serving app.
    Prepare the api for expose to the client by FastAPI router.
    """

    def __init__(self, stock_id: str, start_date: str, end_date: str):
        self._data_fetcher = DataFetcherFactory.create_data_fetcher("yfinance")
        self._stock_analyzer = None
        self._stock_data = None

        try:
            self._data_fetcher.fetch_from_source(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
        except RuntimeError:
            print("Data fetcher failed to fetch data from source")
            return

        self._stock_data = self._data_fetcher.get_as_dataframe()
        if self._stock_data is None:
            print("No stock data available")
            return
        self._stock_analyzer = StockPriceAnalyzer(self._stock_data)

    def calculate_moving_average(self, window_size: int) -> None:
        """
        Calculate the moving average of the company.
        :param window_size:
        :return:
        """
        if self._stock_analyzer is None:
            print("No stock data available")
            return
        self._stock_analyzer.calculating_moving_average(window_size)

    def calculate_daily_return_percentage(self) -> None:
        """Calculate the daily return percentage of the company."""
        if self._stock_analyzer is None:
            return
        self._stock_analyzer.calculating_daily_return_percentage()

    def get_encoded_str_analysis_data(self) -> str:
        """Get the analysis data."""
        if self._stock_analyzer is None:
            raise RuntimeError("No stock data available")

        return json.dumps(self._stock_analyzer.get_analysis_data(), cls=SafeEncoder)


class SafeEncoder(json.JSONEncoder):
    """
    A JSONEncoder that implements the default method to handle NaN and Infinity values.
    """
    def default(self, obj):
        if isinstance(obj, float):
            if obj != obj:  # check for NaN
                return None
            if obj == float('inf'):
                return "Infinity"
            if obj == float('-inf'):
                return "-Infinity"
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return super().default(obj)


