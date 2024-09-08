# pyadps/__main__.py

import subprocess
import os

def main():
    # Get the absolute path to Home_Page.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    home_page_path = os.path.join(current_dir, 'Home_Page.py')

    # Run the Streamlit app
    subprocess.run(["streamlit", "run", home_page_path])

if __name__ == "__main__":
    main()
