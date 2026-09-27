import streamlit as st
import streamlit.components.v1 as components
import os
from src.pipeline import ChatGraphRAGPipeline

st.set_page_config(
    page_title="ChatGraph RAG - Conversational Intelligence Engine",
    layout="wide"
)

st.title("ChatGraph RAG: Conversational Knowledge Graph Engine")
st.markdown("""
A Hybrid Graph & Vector RAG system for Slack & Discord chat histories.
Combines ChromaDB vector search with NetworkX multi-hop knowledge graph traversal.
""")

@st.cache_resource
def load_pipeline():
    p = ChatGraphRAGPipeline()
    p.build_indices()
    return p

with st.spinner("Initializing Knowledge Graph & Vector Store..."):
    pipeline = load_pipeline()

st.sidebar.header("Knowledge Graph Stats")
st.sidebar.metric("Graph Nodes", len(pipeline.graph_mgr.get_nodes()))
st.sidebar.metric("Graph Edges", len(pipeline.graph_mgr.get_edges()))

if st.sidebar.button("Generate & View Graph HTML"):
    html_path = pipeline.graph_mgr.export_html_visualization("graph.html")
    st.sidebar.success("Generated graph.html")

tab1, tab2, tab3 = st.tabs(["Query & Chat", "Interactive Graph Viewer", "Raw Chat Data"])

with tab1:
    st.subheader("Ask Questions About Your Team's Conversations")
    
    example_queries = [
        "What caused the PaymentService 504 error and who fixed it?",
        "What frontend framework did Eve decide on for the customer portal?",
        "Who is assigned to the Presidio PII redaction security ticket?"
    ]
    
    selected_query = st.selectbox("Or choose an example query:", ["Custom Query"] + example_queries)
    
    if selected_query != "Custom Query":
        user_query = selected_query
    else:
        user_query = st.text_input("Enter your question:", "What caused the PaymentService HTTP 504 error?")
        
    if st.button("Submit Query", type="primary"):
        with st.spinner("Executing Hybrid Graph + Vector Search..."):
            res = pipeline.query(user_query)
            
            st.success("Synthesized Answer")
            st.write(res["answer"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Graph Traversal Paths")
                if res["graph_triplets"]:
                    for t in res["graph_triplets"]:
                        st.info(f"**({t['subject']})** ──`{t['relation']}`──▶ **({t['object']})**")
                else:
                    st.write("No graph paths traversed.")
                    
            with col2:
                st.subheader("Vector Text Chunks")
                if res["vector_chunks"]:
                    for c in res["vector_chunks"]:
                        st.caption(f"Distance: {c.get('distance', 0):.2f}")
                        st.text(c["text"])
                else:
                    st.write("No vector chunks fetched.")

with tab2:
    st.subheader("Interactive Knowledge Graph Visualization")
    if os.path.exists("graph.html"):
        with open("graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)
    else:
        if st.button("Render Knowledge Graph"):
            pipeline.graph_mgr.export_html_visualization("graph.html")
            st.rerun()

with tab3:
    st.subheader("Input Chat Transcript")
    chunks = pipeline.data_loader.format_as_chunks()
    for c in chunks:
        st.write(f"**{c['author']}** ({c['timestamp']}): {c['content']}")
