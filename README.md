# How to run:


1. clone the repo:
    ```bash
    git clone https://github.com/tikendraw/ask_me.git

    cd ask_me
    ```

2. rename .env.example to .env and add api key  and model
    ```bash
    mv .env.example .env
    ```

3. create virtual env and activate it
    ```bash
    # install uv if needed
    curl -LsSf https://astral.sh/uv/install.sh | sh  # mac/linux

    uv sync 
    source .venv/bin/activate
    ```

4. be in the ask_me direcotry 
    
    a. run backend
    
    ```bash
    uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
    ```
    

    run server.py
    
    b. run frontend
    ```bash
    streamlit run src.frontend.py
    ```
    

Note: 
1. Rag do not have overall context of entire document, so it will not be able to answer questions that are present on a high level.
2. Rag's performance depends on the model you are using. and the model's ability to understand the context of the document.
3.  Rag's performance can be improved by tuning chunk size and retrieved number of chunks, reranking and more. This implementation does not account that. (it requries time and resource)