from data_preparation import prepare_data
from clustering import run_clustering
from regression import run_regression
from classification import run_classification
from visualization import create_visualizations

def main():
    print("Starting Employee Burnout Prediction pipeline...")
    cleaned_df = prepare_data()
    clustered_df = run_clustering(cleaned_df)

    run_regression()
    run_classification()

    create_visualizations(clustered_df)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()