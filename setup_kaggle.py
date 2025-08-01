"""
Setup script for Kaggle API credentials.
This script helps users set up their Kaggle credentials for dataset access.
"""

import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Interactive setup for Kaggle API credentials"""
    print("🔧 KAGGLE API CREDENTIALS SETUP")
    print("=" * 50)
    print("To download datasets from Kaggle, you need API credentials.")
    print("Follow these steps:")
    print()
    print("1. Go to https://www.kaggle.com/")
    print("2. Sign in to your account (create one if needed)")
    print("3. Go to Account settings (click on your profile picture)")
    print("4. Scroll down to 'API' section")
    print("5. Click 'Create New API Token'")
    print("6. This will download a 'kaggle.json' file")
    print()
    
    # Check if credentials already exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials already found!")
        print(f"Location: {kaggle_json}")
        
        # Verify credentials format
        try:
            with open(kaggle_json, 'r') as f:
                creds = json.load(f)
                if 'username' in creds and 'key' in creds:
                    print("✅ Credentials format is valid")
                    return True
                else:
                    print("❌ Credentials format is invalid")
        except Exception as e:
            print(f"❌ Error reading credentials: {e}")
    
    print("\n📁 Setting up Kaggle credentials directory...")
    
    # Create .kaggle directory
    kaggle_dir.mkdir(exist_ok=True)
    print(f"Created directory: {kaggle_dir}")
    
    # Get credentials from user
    print("\n🔑 Please enter your Kaggle API credentials:")
    print("(You can find these in the downloaded kaggle.json file)")
    
    username = input("Kaggle Username: ").strip()
    api_key = input("Kaggle API Key: ").strip()
    
    if not username or not api_key:
        print("❌ Username and API key are required!")
        return False
    
    # Create credentials file
    credentials = {
        "username": username,
        "key": api_key
    }
    
    try:
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        
        print(f"✅ Credentials saved to: {kaggle_json}")
        print("✅ File permissions set to 600 (secure)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")
        return False

def test_kaggle_connection():
    """Test Kaggle API connection"""
    print("\n🧪 Testing Kaggle API connection...")
    
    try:
        import kagglehub
        
        # Try to access a small public dataset
        print("Attempting to connect to Kaggle API...")
        
        # This will test the connection without downloading large files
        datasets = kagglehub.dataset_list(search="test", page_size=1)
        
        print("✅ Kaggle API connection successful!")
        return True
        
    except ImportError:
        print("❌ kagglehub package not installed")
        print("Run: pip install kagglehub")
        return False
        
    except Exception as e:
        print(f"❌ Kaggle API connection failed: {e}")
        print("\nTroubleshooting:")
        print("• Check your internet connection")
        print("• Verify your Kaggle credentials are correct")
        print("• Make sure your Kaggle account is verified")
        return False

def main():
    """Main setup function"""
    print("🚀 KAGGLE SETUP FOR CUSTOMER CHURN PROJECT")
    print("=" * 60)
    
    # Step 1: Setup credentials
    if setup_kaggle_credentials():
        print("\n" + "="*50)
        
        # Step 2: Test connection
        if test_kaggle_connection():
            print("\n🎉 SETUP COMPLETE!")
            print("You can now run the Kaggle dataset analysis:")
            print("python demo_kaggle.py")
        else:
            print("\n⚠️  Setup completed but connection test failed.")
            print("Please check your credentials and try again.")
    else:
        print("\n❌ Setup failed. Please try again.")
    
    print("\n" + "="*60)
    print("ALTERNATIVE SETUP METHOD:")
    print("If you have the kaggle.json file downloaded:")
    print("1. Create directory: mkdir -p ~/.kaggle")
    print("2. Copy file: cp /path/to/kaggle.json ~/.kaggle/")
    print("3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")

if __name__ == "__main__":
    main()
