# Speaker Recognition System

A robust speaker identification and verification system with liveness detection to prevent replay attacks.

## Features

- **Speaker Enrollment**: Register users with voice samples
- **Speaker Verification**: 1:1 matching (verify claimed identity)
- **Speaker Identification**: 1:N matching (identify unknown speaker)
- **Liveness Detection**: Random sentence generation to prevent replay attacks
- **Speech-to-Text**: Whisper via Groq API for accurate transcription
- **Audio Preprocessing**: Noise reduction, normalization, VAD
- **REST API**: FastAPI endpoints for integration
- **Access Logging**: Track authentication attempts with liveness results


## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/enroll` | POST | Enroll new user |
| `/verify` | POST | Verify speaker identity (with optional liveness) |
| `/identify` | POST | Identify unknown speaker |
| `/users` | GET | List enrolled users |
| `/users/{id}` | GET | Get user details |
| `/users/{id}` | DELETE | Delete user |
| `/logs` | GET | Get access logs |

### Liveness Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/challenge/generate` | POST | Generate liveness challenge |
| `/challenge/{id}` | GET | Get challenge details |


## Project Structure

```
speaker-identification/
├── src/
│   ├── api.py                  # FastAPI endpoints
│   ├── config.py               # Configuration
│   ├── preprocessing.py        # Audio preprocessing
│   ├── embeddings.py           # ECAPA-TDNN embeddings
│   ├── enrollment.py           # User enrollment
│   ├── verification.py         # Speaker verification
│   ├── database.py             # SQLite database
│   ├── sentence_generator.py  # Random sentence generation
│   ├── transcription.py        # Whisper STT via Groq
│   └── liveness.py             # Liveness detection logic
├── tests/                      # Unit tests
├── data/                       # Database storage
├── main.py                     # Entry point
└── requirements.txt
```

## Security Features

- **Replay Attack Prevention**: Single-use challenges with 2-minute expiration
- **Dual Verification**: Voice similarity + sentence matching
- **Fuzzy Matching**: 90% threshold balances security and usability
- **Comprehensive Logging**: All attempts logged with liveness details

## Audio Requirements

- **Format**: WAV, MP3, FLAC, or any format supported by librosa
- **Duration**: 3-10 seconds per sample recommended
- **Quality**: Clear speech, minimal background noise
- **Enrollment**: At least 3 samples per user


