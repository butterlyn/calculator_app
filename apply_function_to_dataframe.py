import pandas as pd
import numpy as np
import logging
from typing import Callable, Union, Optional
from concurrent.futures import ProcessPoolExecutor

# Configure logging settings
logging.basicConfig(filename='log.log', level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s')

def apply_function_to_dataframe(dataframe: pd.DataFrame, func: Callable, num_workers: Optional[int] = None, **func_args) -> Union[pd.Series, pd.DataFrame]:
    """
    Apply the provided function to the input DataFrame using multiprocessing, mapping DataFrame columns to function arguments.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        func (Callable): The function to apply to each row of the dataframe.
        num_workers (Optional[int], optional): The number of parallel workers to use. Defaults to None.
        **func_args: Additional keyword arguments for the function.

    Returns:
        Union[pd.Series, pd.DataFrame]: A Pandas Series or DataFrame containing the results of applying the function to each row of the input dataframe.
    """
    def apply_helper(row):
        try:
            return func(**row[func_args].to_dict())
        except Exception as e:
            logging.warning(f"Error applying function to row: {e}")
            return np.nan

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(apply_helper, [row for _, row in dataframe.iterrows()]))

    return pd.Series(results)

def append_and_output_csv(dataframe: pd.DataFrame, results: Union[pd.Series, pd.DataFrame], filename: str):
    """
    Append the results to the input DataFrame horizontally and output the combined DataFrame as a CSV file.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        results (Union[pd.Series, pd.DataFrame]): The results to append to the dataframe.
        filename (str): The name of the output CSV file.
    """
    combined_dataframe = pd.concat([dataframe, results], axis=1)
    combined_dataframe.to_csv(filename, index=False)
