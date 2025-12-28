To run the project:
1. Clone the repo
   
    git clone https://github.com/BayaderJ/CVProject_TA.git

    cd CVProject_TA

2. Create venv and install dependencies
   
    python -m venv venv

    venv\Scripts\activate 

    pip install -r requirements.txt

3. Run the app (model auto-downloads)
   
    uvicorn app.main:app --reload
