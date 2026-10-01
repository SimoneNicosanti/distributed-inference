from typing import Protocol


class S3Client(Protocol):
    def upload_file(self, filename: str, bucket: str, key: str) -> object: ...

    def download_file(self, bucket: str, key: str, filename: str) -> object: ...

    def head_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
    ) -> object: ...
