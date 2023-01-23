import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import MinMaxScaler


class MarketingCampaignPrediction():
    '''
    A class to represent a predictive maintainance of aircraft.

    ...

    Attributes
    ----------
    model file : file
                predictive model saved after the experiment
            In this case it is the 'Neural Network Regression' model

    Methods
        -------
        load_clean_data(data_file):

        # take a data file (*.txt) and preprocess it
            Import the text file and it process and clean and standardize the file required for prediction
        Parameters

    predicted_vallue():

            Processed data will be predicted.

    predicted_outputs():

         Processed data will be predicted and concated with Original Value.
    pass'''

    def __init__(self, model_files):

        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            model file : file
                predictive model saved after the experiment
            In this case it is the 'Neural Network Regression' model

        """
        # read the 'model' files which were saved
        with open(model_files, 'rb') as model_file:
            self.classification = pickle.load(model_file)

    def load_clean_data(self, data_file):
        # take a data file (*.csv) and preprocess it
        """
            Import the text file and it process and clean and standardize the file required for prediction
        Parameters
        ----------
        data_file : in .txt format

        Returns
        -------
        cleaned and processed file required for prediction

        '''
        """
        # import the data
        df = pd.read_csv(data_file, sep=';')

        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()

        # map Nominal Category variables; the result is a dummy
        df['education'] = df['education'].map({'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})

        df['default'] = df['default'].map({'no': 0, 'yes': 1})

        df['housing'] = df['housing'].map({'no': 0, 'yes': 1})

        df['loan'] = df['loan'].map({'no': 0, 'yes': 1})

        df['month'] = df['month'].map(
            {'dec': 12, 'jan': 1, 'oct': 10, 'jun': 6, 'feb': 2, 'nov': 11, 'apr': 4, 'mar': 3, 'aug': 8,
             'jul': 7, 'may': 5, 'sep': 9})

        df['Actual'] = df['y'].map({'no': 0, 'yes': 1})

        ## One Hot encoding
        df = pd.get_dummies(df, columns=['job', 'marital', 'poutcome'])

        ## Droping the unwanted Column
        df.drop(['day', 'contact', 'y'], axis=1, inplace=True)

        # re-order the columns in df
        Column_names = ['age', 'education', 'default', 'balance', 'housing', 'loan', 'month',
                        'duration', 'campaign', 'pdays', 'previous', 'job_admin.',
                        'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                        'job_management', 'job_retired', 'job_self-employed', 'job_services',
                        'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                        'marital_divorced', 'marital_married', 'marital_single',
                        'poutcome_failure', 'poutcome_other', 'poutcome_success',
                        'poutcome_unknown', 'Actual']

        df = df[Column_names]

        self.preprocessed_data = df.copy()

        df.drop('Actual', axis=1, inplace=True)
        # re-order the columns in df
        Column_names = ['age', 'education', 'default', 'balance', 'housing', 'loan', 'month',
                        'duration', 'campaign', 'pdays', 'previous', 'job_admin.',
                        'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                        'job_management', 'job_retired', 'job_self-employed', 'job_services',
                        'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                        'marital_divorced', 'marital_married', 'marital_single',
                        'poutcome_failure', 'poutcome_other', 'poutcome_success',
                        'poutcome_unknown']

        df = df[Column_names]

        sc = MinMaxScaler()

        self.data = sc.fit_transform(df)

    # a function which outputs the probability of a data point to be 1
    def predicted_vallue(self):
        """
            Processed data will be predicted.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            pred = self.classification.predict(self.data)[:, 1]
            return pred

    # predict the outputs and
    # add columns with these values at the end of the new data

    def predicted_outputs(self):
        """
            Processed data will be predicted and concated with Original Value.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            self.prediction = self.classification.predict(self.data)
            self.preprocessed_data['Prediction'] = self.prediction
            return self.preprocessed_data