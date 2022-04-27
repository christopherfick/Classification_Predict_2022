from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class TrainTestSplit:
    """
    Splits train and test data
    
    Parameters:
        X: Dependant variables
        y: Target variable
        test_size: Indicates size of split for testing and training
        random_state: Random state for reproducible results

    Methods:
        standard_split: basic train_test_split()
        
    """
    def __init__(self, X, y, test_size=.2, random_state=22) -> None:
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def standard_split(self) -> None:
        return train_test_split(self.X, 
                                self.y, 
                                test_size=self.test_size, 
                                random_state=self.random_state)


class ModelLibrary():
    """
    Warehouse for storing different types of models and their parameters

    Methods:
        display_library: Prints library to user

    Returns:
        Model selected from library
    """
    def __init__(self, model_selector=''):
        """
        Initialize and Select model from model_library
        
        Parameters:
            model_selector(str): Key to select model from library

        Attributes:
            model_library(dict): Dictionary in the form of {model_selector: model_to_use}
            model(model): Selected model

        Example:
            >>>ModelLibrary('logistic_reg').model
            LogisticRegression(solver='liblinear')
        """
        self.model_library = {'logistic_reg':LogisticRegression(solver='liblinear'),
                              'count_vect':CountVectorizer(min_df=2, 
                                                           ngram_range=(1,2))}

        self.model_selector = model_selector
        self.model = self.model_library.get(self.model_selector)

        if self.model is None:
            self.display_library()
            raise ValueError(f"Name '{self.model_selector}' not found in library")

    def display_library(self)->None:
        """Displays current library to user"""
        print('---------------------------------------------Start------------------------------------------------------')
        [print(f'model_selctor={key}: {str(value)}') for key, value in self.model_library.items()]
        print('---------------------------------------------End--------------------------------------------------------')
    