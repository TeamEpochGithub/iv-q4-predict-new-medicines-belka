"""Module for the lazy xgboost dataset."""
import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from epochalyst.pipeline.model.training.training_block import TrainingBlock


@dataclass
class LazyXGBDataset:
    """Dataset to load data in batches for xgboost dmatrices.

    :param steps: The training block steps to apply to a batch.
    :param chunk_size: Size of chunk
    """

    steps: list[TrainingBlock]
    chunk_size: int = 10000
    max_queue_size: int = field(default=2, init=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize class."""
        self.prefetch_queue: queue.Queue[xgb.DMatrix | None] = queue.Queue(maxsize=self.max_queue_size)  # to store prefetched matrices
        self.prefetch_thread: threading.Thread | None = None
        self._stop_prefetch = threading.Event()

    def get_iterator(self, X: npt.NDArray[np.string_] | list[str], y: npt.NDArray[np.int_] | list[int]) -> Generator[xgb.DMatrix, None, None]:
        """Get an iterator that yields prefetched DMatrix objects.

        :param X: Input x data
        :param y: Input y data
        :return: An iterator yielding DMatrix objects
        """
        self._start_prefetch_thread(X, y)
        return self._iterator()

    def _calculate_steps(self, X: npt.NDArray[np.string_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.string_], npt.NDArray[np.int_]]:
        """Calculate the data using training steps provided.

        :param X: Input x data
        :param y: Input y data
        :return: Transformed data
        """
        for step in self.steps:
            X, y = step.train(X, y)
        return X, y

    def _prefetch(self, X: npt.NDArray[np.string_], y: npt.NDArray[np.int_]) -> None:
        """Prefetch data and store in a queue for asynchronous access.

        :param X: Input x data
        :param y: Input y data
        """
        index = 0
        while index < len(X) and not self._stop_prefetch.is_set():
            X_subset = X[index : index + self.chunk_size]
            y_subset = y[index : index + self.chunk_size]
            index += self.chunk_size
            x_processed, y_processed = self._calculate_steps(X_subset, y_subset)
            dmatrix = xgb.DMatrix(x_processed, label=y_processed)
            self.prefetch_queue.put(dmatrix)
        self.prefetch_queue.put(None)  # Signal end of data

    def _start_prefetch_thread(self, X: npt.NDArray[np.string_] | list[str], y: npt.NDArray[np.int_] | list[int]) -> None:
        """Start a thread to prefetch data.

        :param X: Input x data
        :param y: Input y data
        """
        self.prefetch_thread = threading.Thread(target=self._prefetch, args=(X, y))
        self.prefetch_thread.start()

    def _iterator(self) -> Generator[xgb.DMatrix, None, None]:
        """Yield prefetched DMatrix objects from the queue.

        :return: A generator yielding DMatrix objects
        """
        while True:
            dmatrix = self.prefetch_queue.get()
            if dmatrix is None:
                break
            yield dmatrix

    def stop_prefetching(self) -> None:
        """Stop the prefetching thread and wait for it to terminate."""
        self._stop_prefetch.set()
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
