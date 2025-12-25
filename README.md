To run the project:
# Clone the repo
git clone https://github.com/BayaderJ/CVProject_TA.git
cd CVProject_TA

# Create venv and install dependencies
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt

# Run the app (model auto-downloads)
uvicorn app.main:app --reload
