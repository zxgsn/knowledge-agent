export const LANGGRAPH_API_URL =
  import.meta.env.VITE_LANGGRAPH_URL || "http://127.0.0.1:2024";

export const LIBRARY_API_BASE = import.meta.env.DEV
  ? import.meta.env.VITE_LIBRARY_API_URL || "http://127.0.0.1:8000"
  : "";
