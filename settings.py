from pydantic_settings import BaseSettings

from llm_tools.lLMInference import LLMInference


class Settings(BaseSettings):
    model_name: str = "IlyaGusev/rugpt_medium_turbo_instructed"
    model_type: str = "gpt2"
    device_type: str = "cpu"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        protected_namespaces = ()


def load_config(llm: LLMInference):
    """
    Load the model and device configuration for an LLMInference instance.

    This function reads configuration settings from environment variables or a .env file (via the Settings class),
    then applies these settings to an instance of LLMInference by loading the specified model and setting the device type.

    Parameters:
    - llm (LLMInference): An instance of the LLMInference class. This object will have its model loaded and device set
                          based on the configuration.

    The function checks for the 'model_name' and 'device_type' settings. If these settings are provided and valid
    (i.e., their string lengths are more than minimal thresholds), it applies them to the LLMInference instance.

    - It loads the model specified by 'model_name' and 'model_type' into the LLMInference instance if a valid
      'model_name' is provided.
    - It sets the device type (e.g., 'cpu' or 'cuda') for the LLMInference instance if a valid 'device_type' is provided.

    Note:
    The function uses a minimal length check (greater than 2 for 'model_name' and greater than 1 for 'device_type')
    as a simple validation step. This ensures that non-empty or meaningful values are provided before attempting
    to load the model or set the device type.
    """
    settings = Settings()  # Initialize the settings object from environment variables or .env file

    if len(settings.device_type) > 1:
        llm.device = settings.device_type
        print(llm.device)

    if len(settings.model_name) > 2:
        llm.load_model(settings.model_name, settings.model_type)

