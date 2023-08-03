# import pytest
# import pandas as pd
# import os
# from unittest import mock
# from src.data_io.data_sinker.local_file_data_sinker import LocalFileDataSinker
#
# # 測試用的 DataFrame
# test_data = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 22],
#     'City': ['New York', 'London', 'Paris']
# })
#
#
# def test_sink_data_csv(tmpdir):
#     # 使用 tmpdir 取得一個臨時目錄作為測試檔案的儲存位置
#     temp_dir = tmpdir.mkdir("test_sink_data_csv")
#
#     # 建立 LocalFileDataSinker 物件
#     data_sinker = LocalFileDataSinker()
#
#     # 使用 patch.object 將 os.makedirs 方法替換成 mock 物件
#     with mock.patch("os.makedirs", side_effect=lambda *args, **kwargs: None):
#         # 呼叫 sink_data 方法
#         destination = os.path.join(temp_dir, "test_data.csv")
#         data_sinker.sink_data(test_data, destination)
#
#         # 確認檔案是否存在
#         assert os.path.exists(destination)
#
#         # 確認檔案內容是否與測試用的 DataFrame 相符
#         loaded_data = pd.read_csv(destination)
#         pd.testing.assert_frame_equal(test_data, loaded_data)
#
#
#
#
# import os
# import pandas as pd
# import pytest
# from unittest.mock import patch, MagicMock
# from src.data_io.data_sinker.local_file_data_sinker import LocalFileDataSinker
#
# # Set up some mock data for the tests
# mock_data = pd.DataFrame({
#     'column1': ['value1', 'value2', 'value3'],
#     'column2': ['value4', 'value5', 'value6']
# })
#
# """
# 這裡的`mock_to_csv`和`mock_exists`分別對應到被模擬的`pandas.DataFrame.to_csv`和`os.path.exists`方法。儘管這兩個參數在函數體內並沒有被直接使用，但是透過設定這些參數，我們在函數內部改變了`pandas.DataFrame.to_csv`和`os.path.exists`的行為。
#
# 具體來說：
#
# - `mock_to_csv`代表模擬的`pandas.DataFrame.to_csv`方法，這樣在單元測試中調用`to_csv`方法時就不會進行真正的IO操作。
# - `mock_exists`則是代表模擬的`os.path.exists`方法，根據設定的返回值（在這裡分別為`True`和`False`），我們可以在單元測試中控制這個方法的行為，從而測試不同路徑存在或不存在時的情況。
#
# 這兩個參數被作為模擬物件（mock objects）傳入測試函數，以便我們可以檢查這些模擬物件的使用情況（例如被調用的次數、調用時的參數等）。這樣我們就可以確認被測試的函數是否正確地使用了這些方法。
# """
#
# @patch('os.path.exists', return_value=False)
# @patch('os.makedirs')
# @patch.object(pd.DataFrame, 'to_csv')
# def test_sink_data_makes_directories_if_not_exist(mock_to_csv, mock_makedirs, mock_exists):
#     sinker = LocalFileDataSinker()
#     sinker.sink_data(mock_data, '/path/to/file.csv')
#     mock_makedirs.assert_called_once_with('/path/to')
#
# @patch('os.path.exists', return_value=True)
# @patch('os.makedirs')
# @patch.object(pd.DataFrame, 'to_csv')
# def test_sink_data_does_not_make_directories_if_exist(mock_to_csv, mock_makedirs, mock_exists):
#     sinker = LocalFileDataSinker()
#     sinker.sink_data(mock_data, '/path/to/file.csv')
#     mock_makedirs.assert_not_called()
#
# def test_sink_data_raises_error_for_unsupported_file_type():
#     sinker = LocalFileDataSinker()
#     with pytest.raises(ValueError):
#         sinker.sink_data(mock_data, '/path/to/file.unsupported')
#
# @patch('os.path.exists', return_value=True)
# @patch('os.makedirs')
# @patch.object(pd.DataFrame, 'to_csv')
# def test_sink_data_csv(mock_to_csv, mock_makedirs, mock_exists):
#     sinker = LocalFileDataSinker()
#     sinker.sink_data(mock_data, '/path/to/file.csv')
#     mock_to_csv.assert_called_once_with('/path/to/file.csv', index=False)
#
# # Similarly, test cases for txt, json and xlsx can be written
#

import pandas as pd
from src.data_io.data_sinker.local_file_data_sinker import LocalFileDataSinker

from unittest.mock import patch
import os


def test_sink_pandas_dataframe():
    # 創建一個示例的 pandas DataFrame
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    destination = "output.csv"

    # 使用 unittest.mock 的 patch 裝飾器來模擬相依物件
    with patch.object(os.path, 'exists', return_value=False), \
         patch.object(os, 'makedirs'), \
         patch('src.data_io.data_sinker.sink_pandas_dataframe.SinkPandasDataFrameFactory.create_sink_adaptor') as mock_create_adaptor:

        # 創建一個偽造的 sink_adaptor 物件，具有 sink 方法，但不會真正寫入任何內容
        class FakeAdaptor:
            def sink(self, data, destination):
                pass

        fake_adaptor = FakeAdaptor()

        # 將 create_sink_adaptor 的返回值設置為 fake_adaptor
        mock_create_adaptor.return_value = fake_adaptor

        # 創建 LocalFileDataSinker 的實例並呼叫 sink_pandas_dataframe 方法
        sinker = LocalFileDataSinker()
        result = sinker.sink_pandas_dataframe(data, destination)

        # 驗證結果
        assert result == True

        # 驗證 create_sink_adaptor 是否用正確的副檔名被調用
        mock_create_adaptor.assert_called_once_with(".csv")

        # 若需驗證其他側效果，也可在此進行



