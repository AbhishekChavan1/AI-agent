import streamlit as st
import asyncio
from pathlib import Path
from agent6 import EmbeddedSystemsAgent

# Streamlit page config
st.set_page_config(page_title="Embedded Systems AI Agent", layout="wide")
st.title("🤖 Embedded Systems AI Agent")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state["agent"] = None

if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""
if "platform" not in st.session_state:
    st.session_state["platform"] = ""
if "projects_list" not in st.session_state:
    st.session_state["projects_list"] = []

# Sidebar: Configuration and Projects
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state["groq_api_key"])
    platform = st.selectbox("Select Platform", ["", "arduino", "esp32", "raspberry_pi"], index=0)
    
    if st.button("Initialize Agent"):
        if not groq_api_key:
            st.warning("Please enter Groq API Key")
        else:
            st.session_state["agent"] = EmbeddedSystemsAgent(groq_api_key)
            st.session_state["groq_api_key"] = groq_api_key
            st.session_state["platform"] = platform
            st.success("Agent initialized!")

    st.markdown("---")
    st.subheader("📁 Projects")
    projects_path = Path("./knowledge_base/projects")
    projects_list = []
    if projects_path.exists():
        projects_list = [p.name for p in projects_path.iterdir() if p.is_dir()]
        st.session_state["projects_list"] = projects_list
    selected_project = st.selectbox("Select a project", [""] + projects_list)

# Ensure agent is initialized
if not st.session_state["agent"]:
    st.warning("Please initialize the agent first.")
    st.stop()
agent = st.session_state["agent"]
st.session_state["platform"] = platform

# Tabs
tab_chat, tab_generate, tab_project, tab_knowledge, tab_history = st.tabs(
    ["💬 Chat", "⚡ Generate Code", "🏗️ Project Builder", "📚 Knowledge Base", "📝 History"]
)

# ------------------ Chat Tab ------------------
with tab_chat:
    st.subheader("💬 Ask anything about Embedded Systems")
    user_input = st.text_area("Your question:", placeholder="E.g., What is ESP32? or How to connect DHT22 sensor?")
    if st.button("Ask", key="chat_btn"):
        if user_input:
            with st.spinner("🤔 Thinking..."):
                result = asyncio.run(agent.process_request(user_input, platform))
            if result["success"]:
                st.success(result["response"])
            else:
                st.error(result["error"])

# ------------------ Code Generation Tab ------------------
with tab_generate:
    st.subheader("⚡ Generate Code")
    requirements = st.text_area("Describe what you want to build:", placeholder="E.g., LED blink with Arduino")
    if st.button("Generate Code", key="generate_btn"):
        if requirements:
            with st.spinner("⚡ Generating code..."):
                result = asyncio.run(agent.process_request(f"Generate {platform} code for: {requirements}", platform))
            if result["success"]:
                st.code(result["response"], language="cpp" if platform in ["arduino", "esp32"] else "python")
            else:
                st.error(result["error"])

# ------------------ Project Builder Tab ------------------
with tab_project:
    st.subheader("🏗️ Generate Full Project")
    project_name = st.text_input("Project Name")
    project_reqs = st.text_area("Project Requirements", placeholder="Describe the project sensors")
    if st.button("Build Project", key="project_btn"):
        if project_name and project_reqs:
            with st.spinner("🏗️ Building project..."):
                result = asyncio.run(agent.generate_project(platform, project_reqs, project_name))
            if result["success"]:
                st.success(f"✅ Project '{project_name}' created!")
                st.json(result)
            else:
                st.error(result["error"])

# ------------------ Knowledge Base Tab ------------------
with tab_knowledge:
    st.subheader("📚 Upload Knowledge Files")
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("📥 Adding to knowledge base..."):
            success = asyncio.run(agent.add_knowledge(file_path))
        if success:
            st.success("File added to knowledge base!")
        else:
            st.error("Failed to add file.")

# ------------------ History Tab ------------------
with tab_history:
    st.subheader("📝 Session History")
    st.info("History is session-based and currently not persistent across reloads.")
    st.write("You can extend Streamlit to save history in `st.session_state` if needed.")

# ------------------ Display Selected Project ------------------
if selected_project:
    st.subheader(f"📂 Project: {selected_project}")
    project_dir = projects_path / selected_project

    project_json = project_dir / "project.json"
    readme_file = project_dir / "README.md"
    code_file = None
    for ext in [".py", ".ino"]:
        candidate = project_dir / f"{selected_project}{ext}"
        if candidate.exists():
            code_file = candidate
            break

    if project_json.exists():
        st.json(project_json.read_text())
    if readme_file.exists():
        st.markdown("### 📄 Documentation")
        st.markdown(readme_file.read_text())
    if code_file and code_file.exists():
        st.markdown("### 💻 Generated Code")
        st.code(code_file.read_text(), language="cpp" if code_file.suffix in [".ino"] else "python")
