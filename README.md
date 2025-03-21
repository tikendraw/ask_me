


HOw to run:


1. clone the repo:
    ```bash
    git clone https://github.com/tikendraw/ask_me.git

    cd ask_me
    ```

2. rename .env.example to .env and add api key  and model
    ```bash
    mv .env.example .env
    ```

3. be in the ask_me direcotry 
    
    a. run backend
    
    ```bash
    uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
    ```
    

    run server.py
    
    b. run frontend
    ```bash
    streamlit run src.frontend.py
    ```
    
