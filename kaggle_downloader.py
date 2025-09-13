# kaggle_downloader.py
# Simplified script to download insurance datasets from Kaggle

import os
import pandas as pd
import zipfile
import subprocess
import sys

def check_kaggle_setup():
    """Check if Kaggle API is set up correctly"""
    print("🔍 Checking Kaggle API setup...")
    
    try:
        # Test if kaggle command works
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Kaggle API is installed and working!")
            return True
        else:
            print("❌ Kaggle API error:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ Kaggle not found. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], 
                         check=True)
            print("✅ Kaggle installed successfully!")
            return True
        except:
            print("❌ Failed to install Kaggle")
            return False
    except Exception as e:
        print(f"❌ Error checking Kaggle: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data",
        os.path.join("data", "raw"), 
        os.path.join("data", "processed")
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def show_available_datasets():
    """Show available insurance datasets"""
    datasets = {
        "1": {
            "name": "rishidamarla/insurance-claim-prediction",
            "description": "Insurance Claim Prediction Dataset - Great for claim amount prediction",
            "good_for": "Predicting claim costs"
        },
        "2": {
            "name": "easonlai/sample-insurance-claim-prediction-dataset",
            "description": "Sample Insurance Claims - Clean and well-structured",
            "good_for": "Learning and prototyping"
        },
        "3": {
            "name": "sagnik1511/car-insurance-data",
            "description": "Car Insurance Dataset - Vehicle-specific data",
            "good_for": "Automotive insurance analysis"
        },
        "4": {
            "name": "noordeen/insurance-premium-prediction",
            "description": "Insurance Premium Dataset - Focus on pricing",
            "good_for": "Premium prediction models"
        },
        "5": {
            "name": "buntyshah/auto-insurance-claims-data",
            "description": "Auto Insurance Claims - Comprehensive dataset",
            "good_for": "Full claims analysis"
        }
    }
    
    print("\n📊 Available Insurance Datasets:")
    print("=" * 60)
    
    for key, dataset in datasets.items():
        print(f"{key}. {dataset['description']}")
        print(f"   📍 Good for: {dataset['good_for']}")
        print(f"   🔗 Dataset: {dataset['name']}")
        print()
    
    return datasets

def download_dataset(dataset_name, choice_num):
    """Download the selected dataset"""
    raw_dir = os.path.join("data", "raw")
    
    print(f"\n📥 Downloading dataset...")
    print(f"Dataset: {dataset_name}")
    print("This might take a few minutes...")
    
    try:
        # Download command
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', raw_dir]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Download failed!")
            print(f"Error: {result.stderr}")
            
            # Common error solutions
            if "403" in result.stderr:
                print("\n💡 This might be because:")
                print("1. Dataset is private or requires competition acceptance")
                print("2. Your Kaggle API token doesn't have permission")
                print("3. You need to accept the dataset's terms on Kaggle website")
            
            return False
        
        print("✅ Download completed!")
        
        # Extract any zip files
        zip_files = [f for f in os.listdir(raw_dir) if f.endswith('.zip')]
        
        for zip_file in zip_files:
            zip_path = os.path.join(raw_dir, zip_file)
            print(f"📦 Extracting {zip_file}...")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_dir)
                
                # Remove zip file after extraction
                os.remove(zip_path)
                print(f"✅ Extracted {zip_file}")
                
            except Exception as e:
                print(f"⚠️ Could not extract {zip_file}: {e}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Download timed out. Please try again.")
        return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def find_csv_files():
    """Find all CSV files in the raw data directory"""
    raw_dir = os.path.join("data", "raw")
    csv_files = []
    
    for file in os.listdir(raw_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(raw_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            csv_files.append({
                'name': file,
                'path': file_path,
                'size_mb': size_mb
            })
    
    return csv_files

def preview_data(csv_files):
    """Preview the downloaded CSV files"""
    if not csv_files:
        print("❌ No CSV files found!")
        return None
    
    print(f"\n📊 Found {len(csv_files)} CSV file(s):")
    print("=" * 50)
    
    for i, file_info in enumerate(csv_files, 1):
        print(f"{i}. {file_info['name']} ({file_info['size_mb']:.1f} MB)")
    
    # Load and preview each file
    dataframes = {}
    
    for file_info in csv_files:
        try:
            print(f"\n📄 Previewing: {file_info['name']}")
            df = pd.read_csv(file_info['path'])
            
            print(f"   📐 Shape: {df.shape}")
            print(f"   📊 Columns: {list(df.columns[:10])}...")  # Show first 10 columns
            
            # Show data types
            print(f"   📋 Sample data:")
            print(df.head(2).to_string())
            
            dataframes[file_info['name']] = df
            
        except Exception as e:
            print(f"   ❌ Error loading {file_info['name']}: {e}")
    
    return dataframes

def find_target_column(dataframes):
    """Automatically suggest target columns"""
    target_keywords = [
        'claim', 'amount', 'cost', 'loss', 'premium', 'price', 
        'fraud', 'total', 'payout', 'expense', 'value'
    ]
    
    suggestions = {}
    
    for filename, df in dataframes.items():
        suggestions[filename] = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for target keywords
            for keyword in target_keywords:
                if keyword in col_lower:
                    # Check if it's numeric
                    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        suggestions[filename].append({
                            'column': col,
                            'reason': f'Contains "{keyword}" and is numeric',
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'mean': df[col].mean()
                        })
                        break
    
    return suggestions

def select_data_for_project(dataframes, suggestions):
    """Help user select the right data for the project"""
    
    # If multiple files, let user choose
    if len(dataframes) > 1:
        print(f"\n🤔 Found {len(dataframes)} CSV files. Which one should we use?")
        files = list(dataframes.keys())
        
        for i, filename in enumerate(files, 1):
            df = dataframes[filename]
            print(f"{i}. {filename} - {df.shape[0]} rows, {df.shape[1]} columns")
        
        while True:
            try:
                choice = int(input("Enter number: ")) - 1
                if 0 <= choice < len(files):
                    selected_file = files[choice]
                    break
                else:
                    print("Please enter a valid number!")
            except ValueError:
                print("Please enter a valid number!")
    else:
        selected_file = list(dataframes.keys())[0]
        print(f"📄 Using: {selected_file}")
    
    df = dataframes[selected_file]
    
    # Select target column
    file_suggestions = suggestions.get(selected_file, [])
    
    if file_suggestions:
        print(f"\n🎯 Suggested target columns for {selected_file}:")
        for i, suggestion in enumerate(file_suggestions, 1):
            print(f"{i}. {suggestion['column']} - {suggestion['reason']}")
            print(f"   Range: ${suggestion['min']:.2f} to ${suggestion['max']:.2f}")
        
        print(f"{len(file_suggestions) + 1}. Let me choose a different column")
        
        while True:
            try:
                choice = int(input("Choose target column: ")) - 1
                if 0 <= choice < len(file_suggestions):
                    target_column = file_suggestions[choice]['column']
                    break
                elif choice == len(file_suggestions):
                    print("Available columns:", list(df.columns))
                    target_column = input("Enter column name: ").strip()
                    if target_column in df.columns:
                        break
                    else:
                        print("Column not found!")
                else:
                    print("Please enter a valid number!")
            except ValueError:
                print("Please enter a valid number!")
    else:
        print(f"\n📋 Available columns in {selected_file}:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        
        target_column = input("Enter target column name: ").strip()
    
    return selected_file, target_column, df

def save_processed_data(df, target_column, original_filename):
    """Save the processed data for the main project"""
    processed_dir = os.path.join("data", "processed")
    
    # Save the main dataset
    output_file = os.path.join(processed_dir, "insurance_data.csv")
    df.to_csv(output_file, index=False)
    
    # Save metadata
    metadata = {
        'original_file': original_filename,
        'target_column': target_column,
        'rows': len(df),
        'columns': len(df.columns),
        'processed_file': output_file
    }
    
    metadata_file = os.path.join(processed_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Data saved to: {output_file}")
    print(f"✅ Metadata saved to: {metadata_file}")
    
    return output_file

def main():
    """Main function"""
    print("🚀 Kaggle Insurance Dataset Downloader")
    print("=" * 50)
    
    # Setup
    setup_directories()
    
    if not check_kaggle_setup():
        print("\n❌ Please set up Kaggle API first!")
        print("\nInstructions:")
        print("1. Go to kaggle.com and create an account")
        print("2. Go to Account → API → Create New API Token")
        print("3. Place the downloaded kaggle.json file in:")
        print("   Windows: C:\\Users\\YourName\\.kaggle\\")
        print("   Mac/Linux: ~/.kaggle/")
        return
    
    # Show datasets
    datasets = show_available_datasets()
    
    # Get user choice
    while True:
        choice = input("Choose dataset (1-5): ").strip()
        if choice in datasets:
            break
        print("Please enter a number between 1-5")
    
    dataset_info = datasets[choice]
    dataset_name = dataset_info["name"]
    
    # Download
    if download_dataset(dataset_name, choice):
        # Find CSV files
        csv_files = find_csv_files()
        
        if csv_files:
            # Preview data
            dataframes = preview_data(csv_files)
            
            if dataframes:
                # Find target suggestions
                suggestions = find_target_column(dataframes)
                
                # Let user select
                selected_file, target_column, df = select_data_for_project(dataframes, suggestions)
                
                # Save for main project
                output_file = save_processed_data(df, target_column, selected_file)
                
                print(f"\n🎉 Success! Dataset ready for analysis!")
                print("=" * 50)
                print(f"✅ File: {output_file}")
                print(f"✅ Target: {target_column}")
                print(f"✅ Shape: {df.shape}")
                print(f"\nNext step: Run 'python run_with_kaggle.py'")
            
        else:
            print("❌ No CSV files found in download")
    
    else:
        print("\n💡 Don't worry! You can still run the project with synthetic data.")
        print("Just run: python run_with_kaggle.py")

if __name__ == "__main__":
    main()