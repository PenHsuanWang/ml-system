import pytest
from src.webapp.data_io_serving_app import DataIOServingApp


def test_singleton_pattern():
    # Get instance of the DataIOServingApp
    app1 = DataIOServingApp.get_app()

    # Get another instance of the DataIOServingApp
    app2 = DataIOServingApp.get_app()

    # They should be the same object (this confirms the singleton pattern)
    assert app1 is app2


def test_thread_safety():
    import threading

    app1 = [None]
    app2 = [None]

    def get_instance(result):
        result[0] = DataIOServingApp.get_app()

    thread1 = threading.Thread(target=get_instance, args=(app1,))
    thread2 = threading.Thread(target=get_instance, args=(app2,))

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    # Check if the two instances are the same
    assert app1[0] is app2[0]