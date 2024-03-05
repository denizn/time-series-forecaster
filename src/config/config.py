"""Module that creates config class and defines functions to get config class"""

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, DirectoryPath
import yaml

# Config file contains the Parameter grid to be used for optimization of models
# These parameters will be tested for multiple sections of the time series with refitting
# The specific cross validation technique (backfitting) will respect temporal nature of time series
# i.e. evaluation dates are later than training sets in the cross validation splits


class Config(BaseModel):
    """
    Leveraging pydantic models to create configs that manage behaviour of the script
    """

    model_config = ConfigDict(title="Config", extra="ignore")

    DATA_PATH: DirectoryPath = Field(default=Path("./data"))
    MODEL_PATH: DirectoryPath = Field(default=Path("./models"))
    REPORT_PATH: DirectoryPath = Field(default=Path("./reports"))
    PARAM_GRID: dict = Field(
        default={
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
            "seasonality_mode": ["additive", "multiplicative"],
        }
    )
    MAX_STORE_COUNT: int = Field(default=None)
    INCLUDE_PROMO: bool = Field(default=True)
    INCLUDE_HOLIDAY: bool = Field(default=True)


def get_config(config_file="./conf/config.yaml"):
    """
    Function loads the config file
    Default location is './conf/config.yaml'
    """

    with open(config_file, "r", encoding="utf-8") as f:
        print(f"""Loading config file: "{config_file}" """)
        config_yaml = yaml.safe_load(f)
    conf = Config(**config_yaml)

    return conf


if __name__ == "__main__":
    get_config()
