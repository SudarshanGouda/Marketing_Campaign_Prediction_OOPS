from PreprocessingCleannig import *

if __name__ == '__main__':

    ### Loading New Data and Predicting

    model = MarketingCampaignPrediction('./finalized_model.sav')

    model.load_clean_data('./train.csv')

    presicted_df = model.predicted_outputs()

    ### Checking the result

    presicted_df = model.predicted_outputs()

    print(presicted_df)

    presicted_df.to_csv('Final_prediction.csv')