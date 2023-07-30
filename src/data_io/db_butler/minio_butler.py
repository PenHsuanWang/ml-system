
from src.db_butler.base_butler import Base

import re
from minio import Minio
from minio.error import InvalidResponseError


class MinioButler(Base):
    """The butler for Minio database."""

    def __init__(self, db_path, user, password, secure=False):
        """Initialize the butler with a path to the database."""
        self.db_path = db_path
        self.client = Minio(
            self.db_path,
            access_key=user,
            secret_key=password,
            secure=secure,
        )

    def get(self, key: str):
        """
        To get the object from desired bucket and filename
        The key contains the information about destination folder and file name
        :param key: <bucket>.<filename>
        :return:
        """

        bucket_name, filename = self.fetch_bucket_name_and_filename_from_key(key)

        return self.client.get_object(bucket_name, filename)

    def put(self, key: str, input_data: str) -> None:
        """
        To put object method define for minio
        the key contains the information about destination folder and file name
        the input data is the path of file which you want to upload
        :param key: <bucket>.<filename>
        :param input_data: path of file
        :return:
        """

        bucket_name, filename = self.fetch_bucket_name_and_filename_from_key(key)

        try:
            self.client.put_object(bucket_name, filename, input_data, len(input_data))
        except InvalidResponseError as e:
            print(f"Failed to put object {filename} into bucket {bucket_name}: {e}")

    def delete(self, key):
        """Delete a value from the database."""
        # TODO: Implement this method
        raise NotImplementedError

    def get_all(self):
        """Get all values from the database."""
        # TODO: Implement this method
        raise NotImplementedError

    def delete_all(self):
        """Delete all values from the database."""
        # TODO: Implement this method
        raise NotImplementedError

    def create_storage(self, storage_name: str):
        """
        Create a bucket.
        :param storage_name: name of the bucket
        :return:
        """
        try:
            self.client.make_bucket(storage_name)
            print(f"Bucket {storage_name} created successfully")
        except InvalidResponseError as e:
            print(f"Failed to create bucket {storage_name}: {e}")

    def storage_exists(self, storage_name: str) -> bool:
        """
        Check if the bucket exists.
        :param storage_name: name of the bucket
        :return: boolean
        """
        return self.client.bucket_exists(storage_name)

    def list_storage(self):
        """
        List all the bucket
        :return:
        """
        return self.client.list_buckets()

    def list_all_storage(self):
        """
        List all the bucket
        :return:
        """
        return self.client.list_buckets()

    def delete_storage(self, storage_name):
        """
        Delete the bucket
        :param storage_name: name of the bucket
        :return:
        """
        # check the bucket exists
        if not self.storage_exists(storage_name):
            print(f"Bucket {storage_name} does not exist")
            return
        try:
            self.client.remove_bucket(storage_name)
            print(f"Bucket {storage_name} deleted successfully")
        except InvalidResponseError as e:
            print(f"Failed to delete bucket {storage_name}: {e}")


    @staticmethod
    def check_key_match_bucket_filename_pattern(key: str) -> bool:
        """
        To check the key match with <bucket>.<filename> pattern
        :param key: <bucket>.<filename>
        :return: boolean
        """
        pattern = re.compile(r"^[a-zA-Z0-9_-]+[.][a-zA-Z0-9_-]+$")
        if not pattern.match(key):
            return False
        return True

    @staticmethod
    def fetch_bucket_name_and_filename_from_key(key: str) -> (str, str):
        """
        To fetch the bucket name and filename from the key
        :param key: <bucket>.<filename>
        :return: list of bucket name and filename
        """
        # check the key match with <bucket>.<filename> pattern
        if not MinioButler.check_key_match_bucket_filename_pattern(key):
            raise ValueError("Key must be in the format <bucket>.<filename>"
                             " where both bucket and filename must be alphanumeric, or - or _")

        key = key.split(".")  # key = ['bucket', 'filename']
        bucket_name = key[0]
        filename = key[1]
        return bucket_name, filename

