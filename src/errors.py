class ApplicationError(Exception):
    default_user_message = "Video processing failed. See the application log for details."

    def __init__(self, message="", user_message=None):
        super().__init__(message)
        self.user_message = user_message or self.default_user_message


class InputValidationError(ApplicationError):
    def __init__(self, message):
        super().__init__(message, message)


class SaveError(InputValidationError):
    pass


class VideoReadError(ApplicationError):
    default_user_message = "The video could not be read to the end."

class VideoWriteError(ApplicationError):
    default_user_message = "The processed video could not be written."

class ModelLoadError(ApplicationError):
    default_user_message = "Required YOLO model is unavailable."

class InferenceError(ApplicationError):
    default_user_message = "Object detection failed."

class TrackerError(ApplicationError):
    default_user_message = "Object tracking failed."

class ConfigurationError(ApplicationError):
    default_user_message = "Invalid application configuration."