import uuid
from typing import Dict, List

import requests
import streamlit as st


class RAGFrontend:
    def __init__(self):
        self.API_URL = "http://localhost:8000"
        
        # Initialize session state
        if 'user_id' not in st.session_state:
            # Generate a new UUID and save it in session state
            st.session_state.user_id = str(uuid.uuid4())
        
        # Use the existing UUID from session state
        self.user_id = st.session_state.user_id

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'urls' not in st.session_state:
            st.session_state.urls = []

    def _clear_data(self) -> bool:
        """Clear ingested data for the user"""
        try:
            response = requests.delete(
                f"{self.API_URL}/clear/{self.user_id}",
            )
            response.raise_for_status()
            # Clear local state
            st.session_state.urls = []
            st.session_state.chat_history = []
            return True
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
            return False

    def render(self):
        st.title("Web Content Q&A System")
        st.caption(f"Session ID: {self.user_id}")
        
        # URL Input Section
        with st.form(key="url_form"):
            url_input = st.text_area("Enter URLs (one per line)", height=100)
            submit_urls = st.form_submit_button("Ingest URLs")
            
            if submit_urls and url_input:
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                st.session_state.urls = urls
                self._ingest_urls(urls)

        if st.button("Clear Data", type="secondary"):
            if self._clear_data():
                st.success("All data cleared successfully!")

        # Query Section
        with st.form(key="query_form"):
            query = st.text_input("Ask a question about the ingested content")
            submit_query = st.form_submit_button("Get Answer")
            
            if submit_query:
                if not query:
                    st.warning("Please enter a question!")
                elif not st.session_state.urls:
                    st.warning("Please ingest some URLs before asking questions!")
                else:
                    answer = self._get_answer(query)
                    if answer:
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer
                        })

        # Display Chat History
        st.subheader("Chat History")
        for item in st.session_state.chat_history[::-1]:
            st.write(f"Q: {item['question']}")
            st.write(f"A: {item['answer']}")
            st.markdown("---")

        # Display Current URLs
        if st.session_state.urls:
            st.subheader("Currently Ingested URLs")
            for url in st.session_state.urls:
                st.write(url)

    def _ingest_urls(self, urls: List[str]) -> None:
        """Send URLs to backend for ingestion"""
        try:
            response = requests.post(
                f"{self.API_URL}/ingest",
                json={"urls": urls},
                headers={"X-User-ID": self.user_id}
            )
            response.raise_for_status()
            st.success("URLs successfully ingested!")
        except Exception as e:
            st.error(f"Error ingesting URLs: {str(e)}")

    def _get_answer(self, query: str) -> str:
        """Get answer from backend"""
        try:
            response = requests.post(
                f"{self.API_URL}/query",
                json={"query": query},
                headers={"X-User-ID": self.user_id}
            )
            
            if response.status_code == 400 and "No data found" in response.json().get("detail", ""):
                st.warning("Please ingest some URLs before asking questions!")
                return None
            
            response.raise_for_status()
            return response.json()["answer"]
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_msg = e.response.json().get('detail', str(e))
                except:
                    pass
            st.error(f"Error getting answer: {error_msg}")
            return None

if __name__ == "__main__":
    app = RAGFrontend()
    app.render()
