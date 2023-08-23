"""
This is the part of code to do stock price analysis
define a class as StockAnalyzer
input the data from YahooFinanceDataLoader in pandas data frame format
"""

import pandas as pd


class StockPriceAnalyzer:
    """Class for analyzing a single company's stock prices."""

    def __init__(self, company_data: pd.DataFrame):
        """
        Initialize with a dataframe of a single company's data.
        :param company_data:
        """
        self._company_data = company_data

    def calculating_moving_average(self, window_size: int):
        """
        Calculate the moving average of the company.
        add new column to the dataframe, column name is "MA_{window_size}_days"
        :param window_size:
        :return:
        """
        column_name = f"MA_{window_size}_days"
        self._company_data[column_name] = self._company_data['Adj Close'].rolling(window_size).mean()

    def calculating_daily_return_percentage(self) -> None:
        """
        Calculate the daily return percentage of the company.
        add new column to the dataframe, column name is "Daily Return"
        :return:
        """
        self._company_data['Daily Return'] = self._company_data['Adj Close'].pct_change()

    def get_analysis_data(self) -> pd.DataFrame:
        """
        Get the analysis data of the company.
        :return: `pd.DataFrame`, the analysis data of the company
        """
        return self._company_data


class CrossStockAnalyzer:

    def __init__(self, companies_data: dict[str, pd.DataFrame]):
        """Initialize with a dictionary of multiple companies' data."""
        self._companies_data = companies_data

    # Add methods that specifically cater to cross analysis of stocks.