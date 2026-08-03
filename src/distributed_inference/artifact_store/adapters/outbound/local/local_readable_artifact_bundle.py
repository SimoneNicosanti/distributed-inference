from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import AsyncGenerator, override

import aiofiles
from aiofiles.threadpool.binary import AsyncBufferedReader

from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    AsyncByteReader,
    ReadableArtifactBundle,
)


class ThreadedFileReader(AsyncByteReader):
    def __init__(self, stream: AsyncBufferedReader) -> None:
        self.stream = stream

    @override
    async def read(self, size: int = -1) -> bytes:
        return await self.stream.read(size)


@dataclass(frozen=True)
class LocalReadableArtifactBundle(ReadableArtifactBundle):
    local_root_path: Path

    @override
    def open_file(
        self, path: PurePosixPath
    ) -> AbstractAsyncContextManager[AsyncByteReader]:
        if path not in self.manifest.get_ppp_set():
            raise FileNotFoundError(f"File is not declared in the artifact: {path}")

        local_file_path = self.local_root_path.joinpath(*path.parts)
        return self._open_local_file(local_file_path)

    @asynccontextmanager
    async def _open_local_file(
        self,
        path: Path,
    ) -> AsyncGenerator[AsyncByteReader]:
        stream = await aiofiles.open(path, "rb")

        try:
            yield ThreadedFileReader(stream)
        finally:
            await stream.close()
