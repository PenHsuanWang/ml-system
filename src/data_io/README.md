# DataIO

DataIO is a Python package providing unified interfaces for various data sources. It offers data fetching and sinking functionalities, supporting a range of data sources such as local files and databases. By using this package, you can easily fetch data from any source and store the processed data into any sink.

## Key Features

- **Data Fetching**: Provides an abstract class `DataFetcher` along with its subclasses such as `LocalFileDataFetcher` and `DatabaseDataFetcher` for reading data from different sources.

- **Data Sinking**: Provides an abstract class `DataSinker` along with its subclasses such as `LocalFileDataSinker` and `DatabaseDataSinker` for storing data into different sinks.

- **Database Operations**: Provides a `DBButler` class and its subclasses like `MySqlButler`, `MinIOButler`, etc., for executing interactive operations with databases.

- **Factory Pattern**: Offers `DataFetcherFactory` and `DataSinkerFactory` that allow you to dynamically create corresponding `DataFetcher` or `DataSinker` instances based on the given data source type.

## How to Use

1. First, create an instance of `DataFetcherFactory` or `DataSinkerFactory`.

```python
fetcher_factory = DataFetcherFactory()
sinker_factory = DataSinkerFactory()
```

2. Then, use the `create_data_fetcher()` or `create_data_sinker()` method of the factory class to create a corresponding `DataFetcher` or `DataSinker` instance based on the provided data source type.

```python
fetcher = fetcher_factory.create_data_fetcher('local_file', '/path/to/file')
sinker = sinker_factory.create_data_sinker('database', 'localhost', db_type='mysql')
```

3. Finally, fetch data with the `fetch_data()` method of `DataFetcher`, or store data with the `sink_data()` method of `DataSinker`.

```python
data = fetcher.fetch_data()
sinker.sink_data(data)
```

4. After you're done, make sure to close the `DataFetcher` or `DataSinker` with their `close()` method to release resources.

```python
fetcher.close()
sinker.close()
```

We hope you enjoy the convenience and efficiency when using DataIO. Any suggestions or issues are always welcome!
