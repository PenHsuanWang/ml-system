from abc import ABC, abstractmethod


class Base(ABC):
    """Abstract base class for all butlers."""

    @abstractmethod
    def __init__(self, db_path):
        """Initialize the butler with a path to the database."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key):
        """Get a value from the database."""
        raise NotImplementedError

    @abstractmethod
    def put(self, key, input_data):
        """Put a value into the database."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key):
        """Delete a value from the database."""
        raise NotImplementedError

    @abstractmethod
    def get_all(self):
        """Get all values from the database."""
        raise NotImplementedError

    @abstractmethod
    def delete_all(self):
        """Delete all values from the database."""
        raise NotImplementedError

    @abstractmethod
    def create_storage(self, storage_name):
        """Create a storage."""
        raise NotImplementedError

    @abstractmethod
    def storage_exists(self, storage_name):
        """Check if storage exists."""
        raise NotImplementedError

    @abstractmethod
    def list_all_storage(self):
        """List all storage."""
        raise NotImplementedError

    @abstractmethod
    def delete_storage(self, storage_name):
        """Delete storage."""
        raise NotImplementedError

