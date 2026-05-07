# Memonic Project Architecture

This document provides a comprehensive overview of the Memonic project structure across its three primary workspaces. It serves as a Context Map to help AI assistants and developers understand the data flow, components, and integration points of the system.

## 🌐 System Architecture

```mermaid
graph LR
    A[Memonic Bracelet] -- Audio Stream --> B[Memonic Backend]
    B -- AI Processed Data --> C[Memonic UI]
    C -- Config/History --> B
```

---

## 📂 Workspaces Breakdown

### 1. ⌚ Memonic Bracelet (Firmware)
**Path:** `/Users/fais/Documents/PlatformIO/Projects/memonic_bracelet`
**Stack:** C++ (PlatformIO framework), ESP32-S3

The hardware component responsible for capturing voice data and streaming it to the backend server.

**Key Files:**
*   `src/main.cpp`: The core application logic. Manages Wi-Fi connections, initializes the I2S interface for the INMP441 microphone, and handles audio streaming via WebSocket/HTTP.
*   `src/secrets.h`: Stores sensitive configuration such as Wi-Fi credentials (SSID/Password) and the target API server URL.
*   `platformio.ini`: PlatformIO configuration file detailing board specifications, upload settings, and required libraries.

### 2. 🧠 Memonic Backend (AI & API)
**Path:** `/Users/fais/Desktop/memonic_backend`
**Stack:** Python, FastAPI, SQLite

The central intelligence of the application. It processes incoming audio streams, transcribes text, extracts meaning/emotion, and stores the user's memories.

**Key Files & Directories:**
*   `ai/api.py`: The main entry point for the FastAPI server. Contains route definitions and WebSocket handlers for receiving audio and serving data to the UI.
*   `ai/memory.py`: Contains the logic for processing transcripts, generating insights, and interfacing with memory storage mechanisms (like Mem0).
*   `ai/models.py`: Defines the database schemas and Pydantic models for data validation.
*   `ai/mcp_server.py`: Manages interactions with external AI agents or services.
*   `requirements.txt`: Lists all Python dependencies required to run the backend (e.g., FastAPI, Uvicorn, OpenAI, Whisper).

### 3. 📱 Memonic UI (Mobile App)
**Path:** `/Users/fais/Desktop/memonic_UI`
**Stack:** React Native, Expo

The user-facing mobile application that displays processed memories, voice history, and allows configuration of the bracelet.

**Key Files & Directories:**
*   `app/index.js`: The main dashboard or entry screen of the application.
*   `app/signin.js`: Handles user authentication and session management.
*   `app/Voice_History.js`: Displays a timeline or list of past audio recordings and their transcriptions.
*   `app/member.js`: Manages user profiles or team settings.
*   `app/config.js`: Contains application settings, crucially the backend API URL.
*   `components/`: A directory for reusable UI elements.

---

## 💡 Key Integration Points

1.  **Audio Pipeline:** The ESP32 (`main.cpp`) streams raw audio data directly to the Python backend (`api.py`).
2.  **Data Synchronization:** The UI (`Voice_History.js`) fetches the processed text, insights, and emotions from the backend (`api.py`) via REST APIs.
3.  **Network Configuration:** For the system to work locally, the API endpoint specified in the UI (`config.js`) and the target URL in the ESP32 (`secrets.h`) must point to the same local IP address of the machine hosting the backend.
