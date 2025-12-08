# Speaker Identification System

A Python-based speaker verification system using deep learning embeddings for biometric authentication.

## Features

- **Audio Preprocessing**: Resampling, noise reduction, VAD, loudness normalization
- **Speaker Embeddings**: ECAPA-TDNN model via SpeechBrain (192-dim vectors)
- **Enrollment**: Register speakers with multiple audio samples
- **Verification**: 1:1 identity verification with cosine similarity
- **Identification**: 1:N speaker search across enrolled users
- **REST API**: FastAPI endpoints for integration
- **Access Logging**: Track authentication attempts

## Installation

```bash
# Clone repository
git clone <repository-url>
cd speaker-identification

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### CLI Usage

```bash
# Enroll a new user (minimum 3 samples recommended)
python main.py enroll --user john --name "John Doe" --audio samples/john1.wav samples/john2.wav samples/john3.wav

# Verify speaker identity
python main.py verify --user john --audio test.wav

# Identify unknown speaker
python main.py identify --audio unknown.wav --top 3

# List enrolled users
python main.py list

# Delete user
python main.py delete --user john
```

### API Server

```bash
# Start server
python main.py serve --port 8000

# Or with auto-reload for development
python main.py serve --port 8000 --reload
```

Access Swagger UI at: http://localhost:8000/docs

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/enroll` | POST | Enroll new user with audio files |
| `/verify` | POST | Verify speaker identity |
| `/identify` | POST | Identify unknown speaker |
| `/users` | GET | List enrolled users |
| `/users/{id}` | GET | Get user details |
| `/users/{id}` | DELETE | Delete user |
| `/logs` | GET | Get access logs |

## Configuration

Edit `src/config.py` to customize:

```python
# Audio settings
sample_rate: int = 16000
target_lufs: float = -23.0

# Verification settings
similarity_threshold: float = 0.5  # Adjust for security/convenience tradeoff
min_enrollment_samples: int = 3

# Model settings
device: str = "cuda"  # or "cpu"
```

## Project Structure

```
speaker-identification/
├── src/
│   ├── config.py         # Configuration
│   ├── preprocessing.py  # Audio preprocessing
│   ├── embeddings.py     # ECAPA-TDNN embeddings
│   ├── database.py       # SQLite storage
│   ├── enrollment.py     # User enrollment
│   ├── verification.py   # Speaker verification
│   └── api.py            # REST API
├── tests/                # Unit tests
├── data/                 # Database files
├── main.py               # CLI entry point
└── requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Audio Requirements

- **Format**: WAV, MP3, FLAC, or any format supported by librosa
- **Duration**: 3-10 seconds per sample recommended
- **Quality**: Clear speech, minimal background noise
- **Samples**: At least 3 samples per user for enrollment

## Threshold Tuning

The verification threshold (default: 0.5) controls the trade-off:

| Threshold | Security | False Rejections |
|-----------|----------|------------------|
| 0.7+ | High | More frequent |
| 0.5 | Balanced | Moderate |
| 0.3- | Low | Rare |

Adjust based on your security requirements.

## License

MIT License
