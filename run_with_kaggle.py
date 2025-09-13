# run_with_kaggle.py
import os
import pandas as pd
from insurance_predictor import InsuranceClaimsPredictor

def main():
    print("🚀 Insurance Claims Predictor")
    print("=" * 40)
    
    # Check if we have Kaggle data
    kaggle_data_path = os.path.join("data", "processed", "insurance_data.csv")
    
    if os.path.exists(kaggle_data_path):
        print("✅ Found Kaggle data!")
        use_kaggle = input("Use Kaggle data? (y/n): ").lower().startswith('y')
        
        if use_kaggle:
            predictor = InsuranceClaimsPredictor()
            # Load the Kaggle data
            predictor.df = pd.read_csv(kaggle_data_path)
            print(f"📊 Loaded Kaggle dataset: {predictor.df.shape}")
        else:
            predictor = InsuranceClaimsPredictor()
            predictor.load_and_explore_data()
    else:
        print("❌ No Kaggle data found. Using synthetic data.")
        print("💡 Run 'python kaggle_downloader.py' first to get real data!")
        predictor = InsuranceClaimsPredictor()
        predictor.load_and_explore_data()
    
    # Run the full analysis
    print("\n🔧 Preprocessing data...")
    predictor.preprocess_data()
    
    print("\n⚙️ Preparing features...")
    predictor.prepare_features()
    
    print("\n🤖 Training models...")
    predictor.train_models()
    
    print("\n📊 Evaluating models...")
    predictor.evaluate_models()
    
    print("\n🎯 Feature importance...")
    predictor.feature_importance_analysis()
    
    print("\n📈 Prediction analysis...")
    predictor.prediction_analysis()
    
    print("\n📝 Final report...")
    predictor.generate_report()
    
    print("\n🎉 Complete!")

if __name__ == "__main__":
    main()