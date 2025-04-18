# ✨ Experience and Learnings

## 🌟 What Worked Well

- **Modular Pipeline**: Separating tasks (`pdf_parser`, `timeline`, `categorizer`, `visualizer`, `storyteller`) with Prefect ensured scalability and maintainability.
- **FastAPI and Streamlit**: Enabled rapid development of a robust API and interactive UI.
- **LLM Integration**: Ollama's storytelling added engaging, user-friendly financial narratives.
- **MLflow Logging**: Tracked artifacts and metrics, simplifying debugging and monitoring.

## 🧩 Challenges

- **PDF Parsing**: Handling diverse bank statement formats required extensive preprocessing and validation.
- **Pydantic Serialization**: JSON serialization issues with Pydantic models (e.g., `CustomerInfo`, `VisualizerOutput`) demanded careful handling.
- **LLM Consistency**: Ensuring reliable LLM outputs required robust prompt engineering and error handling.
- **Data Validation**: Managing `NaN` values and column mismatches in CSVs was time-consuming.

## 💡 What We Learned

- **Pipeline Orchestration**: Prefect simplified complex workflows, but task dependencies need clear definitions.
- **Pydantic Best Practices**: Always use `.dict()` for JSON serialization to avoid type errors.
- **LLM Prompting**: Specific, context-rich prompts improve narrative quality.
- **Testing Early**: Unit tests for data validation and serialization catch errors before runtime.

## 🔍 Project Difficulty

This project was **moderately complex**, balancing PDF parsing, data pipelines, AI integration, and UI development. The result is a scalable platform for financial health analysis.

---

## Next Steps

- [Installation Guide](installation.md)
- [Usage Guide](usage.md)