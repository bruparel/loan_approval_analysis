
import pandas as pd

def load_data(file_path):
    """Load loan data from a specified file path."""
    data = pd.read_excel(file_path)
    data.set_index('Loan_ID', inplace=True)
    return data
