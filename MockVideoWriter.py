class MockVideoWriter:
    """
    A mock video writer class to store the written frame.
    It simulates the behavior of cv2.VideoWriter for testing and debugging purposes.
    """

    def __init__(self):
        """
        Initialize the MockVideoWriter with a placeholder for storing frames.
        """
        self.frame = None  # Store the frame that is written

    def write(self, frame):
        """
        Save the given frame to the instance variable.

        Args:
            frame: The frame to be written (stored).
        """
        self.frame = frame  # Save the frame to the instance variable

    def release(self):
        """
        Placeholder method to simulate the release of resources.
        No action is needed for this mock class.
        """
        pass
