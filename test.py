# from fpdf import FPDF

# pdf = FPDF()
# pdf.add_page()
# pdf.set_font("Arial", size=12)
# text_string = "This is the string to be saved as PDF."
# pdf.cell(200, 10, txt=text_string, ln=True, align='C')
# pdf.output("string_as_pdf.pdf")

# import spacy
# import pandas as pd
# import networkx as nx
# from itertools import combinations

# def extract_nodes_edges(text):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
    
#     # Extract meaningful entities (nodes)
#     nodes = set()
#     edges = []
    
#     prev_entity = None
    
#     for ent in doc.ents:
#         nodes.add(ent.text)
        
#         if prev_entity:
#             edges.append((prev_entity, ent.text))
        
#         prev_entity = ent.text
    
#     # Create DataFrames
#     nodes_df = pd.DataFrame({'Node': list(nodes)})
#     edges_df = pd.DataFrame(edges, columns=['Source', 'Target'])
    
#     return nodes_df, edges_df

# # Example response text
# response_text = """
# Here are the key differences between the EMBED (EMory BrEast imaging Dataset) and DMID (Digital mammography dataset for breast cancer diagnosis research) datasets:

# **EMBED**:
# - **Data Availability**: Available upon signing an agreement.
# - **Data Link**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview
# - **Associated Article**: The EMory BrEast imaging Dataset (EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic Mammographic Images (published on: 2023)
# - **Types of data**: Mammogram_Images
# - **Types of files**: dcm
# - **Data collected from**: USA
# - **Name**: EMBED
# - **Information**: Stands for EMory BrEast imaging Dataset. This dataset contains approximately 3.4 million screening and diagnostic mammographic images.

# **DMID**:
# - **Data Availability**: Accessible
# - **Data Link**: https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883
# - **Associated Article**: Digital mammography dataset for breast cancer diagnosis research (DMID) with breast mass segmentation analysis (published on: 2024)
# - **Types of data**: Mammogram_Images, Metadata
# - **Types of files**: dcm, xlsx, txt
# - **Data collected from**: India
# - **Name**: DMID
# - **Information**: This dataset contains a total of 510 images. Among them, 274 are abnormal images, including benign and malignant cases.

# **Differences**:
# 1. **Size**: EMBED is significantly larger (~3.4 million images) compared to DMID (510 images).
# 2. **Origin**: EMBED was collected from the USA, while DMID was collected from India.
# 3. **Data Availability**: EMBED requires signing an agreement for access, whereas DMID is accessible without any restrictions.
# 4. **Associated Article**: The articles associated with each dataset were published in different years (2023 for EMBED and 2024 for DMID).
# 5. **File Types**: While both datasets contain dcm files, DMID also includes xlsx and txt files for metadata.
# 6. **Purpose**: Although both can be used for breast cancer detection, the larger size and racially diverse nature of EMBED make it suitable for more granular analysis, while DMID might be better suited for specific tasks like mass segmentation due to its associated article's focus on that topic.
# """

# nodes_df, edges_df = extract_nodes_edges(response_text)

# print("Nodes DataFrame:")
# print(nodes_df)
# print("\nEdges DataFrame:")
# print(edges_df)

# import spacy
# import pandas as pd
# import networkx as nx
# from itertools import combinations

# # Load spaCy model for Named Entity Recognition (NER)
# nlp = spacy.load("en_core_web_trf")

# def extract_entities(text):
#     """Extracts meaningful entities (nodes) from the text."""
#     doc = nlp(text)
#     entities = [ent.text for ent in doc.ents]  # Extract named entities
#     if not entities:  # If no named entities, use noun chunks as backup
#         entities = [chunk.text for chunk in doc.noun_chunks]
#     return list(set(entities))  # Remove duplicates

# def generate_edges(nodes):
#     """Generates edges between nodes (if they co-occur in the same text)."""
#     edges = list(combinations(nodes, 2))  # Create pairs of nodes
#     return edges

# def create_graph_dataframe(text):
#     """Processes LLM response, extracts nodes and edges, and creates a dataframe."""
#     nodes = extract_entities(text)
#     edges = generate_edges(nodes)
    
#     df = pd.DataFrame(edges, columns=["Node1", "Node2"])
#     return df

# # Example LLM response
# llm_response = """
# # Here are the key differences between the EMBED (EMory BrEast imaging Dataset) and DMID (Digital mammography dataset for breast cancer diagnosis research) datasets:

# # **EMBED**:
# # - **Data Availability**: Available upon signing an agreement.
# # - **Data Link**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview
# # - **Associated Article**: The EMory BrEast imaging Dataset (EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic Mammographic Images (published on: 2023)
# # - **Types of data**: Mammogram_Images
# # - **Types of files**: dcm
# # - **Data collected from**: USA
# # - **Name**: EMBED
# # - **Information**: Stands for EMory BrEast imaging Dataset. This dataset contains approximately 3.4 million screening and diagnostic mammographic images.

# # **DMID**:
# # - **Data Availability**: Accessible
# # - **Data Link**: https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883
# # - **Associated Article**: Digital mammography dataset for breast cancer diagnosis research (DMID) with breast mass segmentation analysis (published on: 2024)
# # - **Types of data**: Mammogram_Images, Metadata
# # - **Types of files**: dcm, xlsx, txt
# # - **Data collected from**: India
# # - **Name**: DMID
# # - **Information**: This dataset contains a total of 510 images. Among them, 274 are abnormal images, including benign and malignant cases.

# # **Differences**:
# # 1. **Size**: EMBED is significantly larger (~3.4 million images) compared to DMID (510 images).
# # 2. **Origin**: EMBED was collected from the USA, while DMID was collected from India.
# # 3. **Data Availability**: EMBED requires signing an agreement for access, whereas DMID is accessible without any restrictions.
# # 4. **Associated Article**: The articles associated with each dataset were published in different years (2023 for EMBED and 2024 for DMID).
# # 5. **File Types**: While both datasets contain dcm files, DMID also includes xlsx and txt files for metadata.
# # 6. **Purpose**: Although both can be used for breast cancer detection, the larger size and racially diverse nature of EMBED make it suitable for more granular analysis, while DMID might be better suited for specific tasks like mass segmentation due to its associated article's focus on that topic.
# # """

# # Create the graph dataframe
# df_graph = create_graph_dataframe(llm_response)
# print(df_graph)
# # print(df_graph['Node2'].unique())

# Optional: Create a graph using networkx
# G = nx.from_pandas_edgelist(df_graph, "Node1", "Node2")
# nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")


import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_meaningful_nodes(text, top_n=5):
    """Extract key phrases as nodes using sentence similarity."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]  # Split into sentences and clean up
    if len(sentences) == 0:
        return []  # If no sentences found, return empty list

    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute similarity between all pairs
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # Select the top N most distinct sentences as key nodes
    scores = similarity_matrix.mean(dim=1)  # Average similarity per sentence
    top_indices = scores.argsort(descending=True)[:min(top_n, len(sentences))]  # Prevent out-of-range

    nodes = [sentences[i] for i in top_indices]
    
    print("Extracted Nodes:", nodes)  # Debugging: Print extracted nodes
    return list(set(nodes))  # Remove duplicates

def generate_edges(nodes):
    """Creates edges based on co-occurrence of nodes."""
    if len(nodes) < 2:
        return []  # No edges possible if fewer than 2 nodes
    edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    return edges

def create_graph_dataframe(text):
    """Processes text, extracts nodes and edges, and creates a DataFrame."""
    nodes = extract_meaningful_nodes(text)
    
    if not nodes:
        print("No meaningful nodes found.")  # Debugging output
        return pd.DataFrame(columns=["Node1", "Node2"])  # Return empty DataFrame

    edges = generate_edges(nodes)

    if not edges:
        print("Only one node found, no edges can be formed.")  # Debugging output
        return pd.DataFrame(columns=["Node1", "Node2"])  # Return empty DataFrame

    df = pd.DataFrame(edges, columns=["Node1", "Node2"])
    return df

# Example LLM Response (Try changing this text to test different cases)
llm_response = """
Here are the key differences between the EMBED (EMory BrEast imaging Dataset) and DMID (Digital mammography dataset for breast cancer diagnosis research) datasets:

**EMBED**:
- **Data Availability**: Available upon signing an agreement.
- **Data Link**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview
- **Associated Article**: The EMory BrEast imaging Dataset (EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic Mammographic Images (published on: 2023)
- **Types of data**: Mammogram_Images
- **Types of files**: dcm
- **Data collected from**: USA
- **Name**: EMBED
- **Information**: Stands for EMory BrEast imaging Dataset. This dataset contains approximately 3.4 million screening and diagnostic mammographic images.

**DMID**:
- **Data Availability**: Accessible
- **Data Link**: https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883
- **Associated Article**: Digital mammography dataset for breast cancer diagnosis research (DMID) with breast mass segmentation analysis (published on: 2024)
- **Types of data**: Mammogram_Images, Metadata
- **Types of files**: dcm, xlsx, txt
- **Data collected from**: India
- **Name**: DMID
- **Information**: This dataset contains a total of 510 images. Among them, 274 are abnormal images, including benign and malignant cases.

**Differences**:
1. **Size**: EMBED is significantly larger (~3.4 million images) compared to DMID (510 images).
2. **Origin**: EMBED was collected from the USA, while DMID was collected from India.
3. **Data Availability**: EMBED requires signing an agreement for access, whereas DMID is accessible without any restrictions.
4. **Associated Article**: The articles associated with each dataset were published in different years (2023 for EMBED and 2024 for DMID).
5. **File Types**: While both datasets contain dcm files, DMID also includes xlsx and txt files for metadata.
6. **Purpose**: Although both can be used for breast cancer detection, the larger size and racially diverse nature of EMBED make it suitable for more granular analysis, while DMID might be better suited for specific tasks like mass segmentation due to its associated article's focus on that topic.
"""

# Generate graph DataFrame
df_graph = create_graph_dataframe(llm_response)
print(df_graph)

# Create and visualize graph
if not df_graph.empty:
    G = nx.from_pandas_edgelist(df_graph, "Node1", "Node2")
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
else:
    print("Graph is empty. No edges to visualize.")

