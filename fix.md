# Memonic — Fix List

Prioritised checklist from the May 2026 code review. Check off each item as it is done.

---

## 🔴 Critical — Fix Before Anyone Else Uses This

- [ ] **[backend] Plaintext passwords**
  `core/models.py` stores raw password strings and `core/main.py` compares them directly.
  Install `passlib[bcrypt]`, hash on account creation, verify on login.
  **Files:** `core/models.py`, `core/main.py`

- [ ] **[bracelet] WiFi credentials hardcoded in source**
  `ssid = "Fais"` and `password = "12345678"` are committed to the repo.
  Create `src/secrets.h`, add it to `.gitignore`, and `#include` it from `main.cpp`.
  **Files:** `src/main.cpp`, `src/secrets.h` (new), `.gitignore`

- [ ] **[bracelet] `STREAM` / `STOP_STREAM` commands are silently ignored**
  `onWsEvent` has no branch for `"STREAM"` or `"STOP_STREAM"` → stream mode never
  starts on the device even when the server sends the command.
  Add the missing `else if` branches in `onWsEvent`.
  **Files:** `src/main.cpp`

- [ ] **[backend] `/api/account/{user_name}` DELETE has no auth guard**
  Anyone can delete any account with zero credentials. Add a password-confirmation
  body param and verify it before deleting.
  **Files:** `core/main.py`

---

## 🟠 High — Fix Soon

- [ ] **[backend] Duplicate / conflicting route definitions**
  `core/main.py` defines placeholder versions of six routes that return hardcoded dummy
  data (`"mood": "happy"`, `"battery": 85` etc.). The real implementations live in
  `ai/api.py` and are already mounted. Delete the duplicates from `core/main.py`.
  **Routes to remove:** `/api/members`, `/api/get-mood/{user_id}`, `/api/get-home-data/{user_id}`,
  `/api/get-events/{user_id}`, `/device-status`, `/api/check-popup/{user_id}`
  **Files:** `core/main.py`

- [ ] **[backend] `ollama` pinned to 0.1.7 but code uses the 0.2+ API**
  `memory.py` calls `ollama.embed(...)` which only exists in `ollama >= 0.2.0`.
  Update the pin: `ollama>=0.2.0`.
  **Files:** `requirements.txt`

- [ ] **[backend] Model init race condition at import time**
  `ai/api.py` calls `asyncio.get_event_loop()` at module import to schedule
  `init_models()`. This raises a `DeprecationWarning` (soon an error) in Python 3.10+
  when called outside an async context, and causes double-initialization.
  Remove the import-time block entirely; the `@app.on_event("startup")` handler in
  `core/main.py` already covers this.
  **Files:** `ai/api.py`

- [ ] **[backend] `/api/memories` ignores `user_id` filter**
  The endpoint returns ALL memories for ALL users regardless of the `user_id` query param
  the UI sends. Add `.filter(Memory.speaker == user_id)` when the param is present.
  **Files:** `ai/api.py`

- [ ] **[ui] Remove dead BLE dependency**
  The bracelet now uses WebSocket, not BLE. The `react-native-ble-plx` package,
  `BLEContext.js`, `useMemonicBLE.js`, and the hook import in `app/index.js`
  are all unused and add unnecessary native permissions on iOS/Android.
  **Files:** `package.json`, `app/index.js`, `context/BLEContext.js`, `hooks/useMemonicBLE.js`

---

## 🟡 Medium — Clean Up

- [ ] **[backend] Add JWT / session auth middleware**
  After login, issue a signed JWT. Require it as a Bearer token on all `/api/*` endpoints.
  Use `python-jose` + the already-present `passlib`.

- [ ] **[backend] Whisper language hardcoded to English**
  `ai/models.py` passes `language="en"` to `_whisper.transcribe(...)`. Remove it (or make
  it a config param) so Whisper auto-detects the speaker's language.
  **Files:** `ai/models.py`

- [ ] **[backend] `@app.on_event("startup")` is deprecated**
  Replace with a `lifespan` async context manager.
  **Files:** `core/main.py`

- [ ] **[backend] Dead standalone app in `ai/api.py`**
  The `app = FastAPI()` + `app.include_router(router)` lines at the bottom of `ai/api.py`
  create a second FastAPI instance that is never served. Remove them.
  **Files:** `ai/api.py`

- [ ] **[backend] `delete_account` does not cascade**
  Deleting a user leaves orphaned `ChatMessage` and `Memory` rows. Delete related rows
  before (or via cascade on) the user delete.
  **Files:** `core/main.py`

- [ ] **[ui] Login button has no loading / disabled state**
  `app/index.js` login button can be double-tapped, firing two concurrent requests.
  Add `loading` state + `disabled={loading}` (same pattern as `signin.js`).
  **Files:** `app/index.js`

- [ ] **[ui] Dead "Login with passkey" button**
  `app/index.js` has a `TouchableOpacity` with no `onPress`. Wire it up or remove it.
  **Files:** `app/index.js`

---

## 🟢 Low — Nice to Have

- [ ] **[backend] Remove unused `mem0ai` dependency**
  `mem0ai==0.0.5` is in `requirements.txt` but is not imported anywhere (switched to ChromaDB).
  **Files:** `requirements.txt`

- [ ] **[bracelet] Consider AGC or softer gain**
  The fixed `x8` gain in `i2sTask` saturates to ±32767 in loud environments.
  A simple peak-normalised gain per chunk would give cleaner audio.
  **Files:** `src/main.cpp`

- [ ] **[bracelet] SSL certificate verification is disabled**
  `beginSslWithCA(..., NULL)` skips cert verification (encrypted but not authenticated).
  If you control the server cert, pin it.
  **Files:** `src/main.cpp`
