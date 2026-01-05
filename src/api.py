"""FastAPI REST API for speaker identification system."""

import os
import tempfile
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .enrollment import (
    enroll_user, 
    update_enrollment, 
    delete_enrollment, 
    list_enrolled_users,
    EnrollmentError
)
from .verification import verify_speaker, identify_speaker
from .database import get_database
from .liveness import generate_challenge, LivenessError


app = FastAPI(
    title="Speaker Recognition API",
    description="REST API for speaker enrollment and verification",
    version="1.0.0"
)

# Add CORS middleware to allow Streamlit UI requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Response Models ====================

class EnrollmentResponse(BaseModel):
    user_id: str
    name: str
    num_samples: int
    status: str


class VerificationResponse(BaseModel):
    user_id: str
    is_verified: bool
    score: float
    threshold: float
    decision: str
    liveness_result: Optional[dict] = None


class UserInfo(BaseModel):
    id: str
    name: str
    num_samples: int
    created_at: str
    updated_at: str


class IdentifyResult(BaseModel):
    user_id: str
    name: str
    score: float


class ChallengeResponse(BaseModel):
    challenge_id: str
    sentence: str
    expires_at: str
    user_id: Optional[str] = None


# ==================== Helper Functions ====================

async def save_upload_files(files: List[UploadFile]) -> List[str]:
    """Save uploaded files to temp directory and return paths."""
    temp_dir = tempfile.mkdtemp()
    paths = []
    
    for file in files:
        path = os.path.join(temp_dir, file.filename)
        with open(path, 'wb') as f:
            content = await file.read()
            f.write(content)
        paths.append(path)
    
    return paths


def cleanup_temp_files(paths: List[str]):
    """Remove temporary files."""
    if paths:
        temp_dir = os.path.dirname(paths[0])
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== Endpoints ====================

@app.post("/enroll", response_model=EnrollmentResponse)
async def enroll(
    user_id: str = Form(...),
    name: str = Form(...),
    audio_files: List[UploadFile] = File(...)
):
    """
    Enroll a new user with audio samples.
    
    - user_id: Unique identifier for the user
    - name: Display name
    - audio_files: At least 3 audio samples (WAV, MP3, etc.)
    """
    paths = []
    try:
        paths = await save_upload_files(audio_files)
        result = enroll_user(user_id, name, paths)
        return EnrollmentResponse(**result)
    except EnrollmentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files(paths)


@app.post("/enroll/{user_id}/update", response_model=EnrollmentResponse)
async def update_user_enrollment(
    user_id: str,
    audio_files: List[UploadFile] = File(...),
    replace: bool = Form(False)
):
    """
    Update existing user's enrollment with new samples.
    
    - **replace**: If true, replace existing embedding; otherwise combine
    """
    paths = []
    try:
        paths = await save_upload_files(audio_files)
        result = update_enrollment(user_id, paths, replace=replace)
        return EnrollmentResponse(
            user_id=result['user_id'],
            name="",  # Not returned by update
            num_samples=result['num_samples'],
            status=result['status']
        )
    except EnrollmentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cleanup_temp_files(paths)


@app.post("/verify", response_model=VerificationResponse)
async def verify(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    challenge_id: Optional[str] = Form(None)
):
    """
    Verify if audio matches claimed speaker identity.
    
    - **user_id**: Claimed speaker ID
    - **audio_file**: Test audio sample
    - **threshold**: Optional custom threshold (default: 0.7)
    - **challenge_id**: Optional challenge ID for liveness detection
    """
    paths = []
    try:
        paths = await save_upload_files([audio_file])
        result = verify_speaker(
            paths[0], 
            user_id, 
            threshold=threshold,
            challenge_id=challenge_id
        )
        return VerificationResponse(
            user_id=result.user_id,
            is_verified=result.is_verified,
            score=result.score,
            threshold=result.threshold,
            decision=result.decision,
            liveness_result=result.liveness_result
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LivenessError as e:
        raise HTTPException(status_code=400, detail=f"Liveness check failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files(paths)


@app.post("/identify", response_model=List[IdentifyResult])
async def identify(
    audio_file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    top_n: int = Form(5)
):
    """
    Identify speaker from enrolled users (1:N matching).
    
    - **audio_file**: Test audio sample
    - **threshold**: Minimum similarity threshold
    - **top_n**: Number of top matches to return
    """
    paths = []
    try:
        paths = await save_upload_files([audio_file])
        results = identify_speaker(paths[0], threshold=threshold, top_n=top_n)
        return [IdentifyResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files(paths)


@app.get("/users", response_model=List[UserInfo])
async def list_users():
    """List all enrolled users."""
    users = list_enrolled_users()
    return [UserInfo(**u) for u in users]


@app.get("/users/{user_id}", response_model=UserInfo)
async def get_user(user_id: str):
    """Get user details by ID."""
    db = get_database()
    user = db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return UserInfo(
        id=user.id,
        name=user.name,
        num_samples=user.num_samples,
        created_at=user.created_at.isoformat(),
        updated_at=user.updated_at.isoformat()
    )


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete user enrollment."""
    if delete_enrollment(user_id):
        return {"status": "deleted", "user_id": user_id}
    raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")


@app.get("/logs")
async def get_logs(user_id: Optional[str] = None, limit: int = 50):
    """Get access logs."""
    db = get_database()
    logs = db.get_access_logs(user_id=user_id, limit=limit)
    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "timestamp": log.timestamp.isoformat(),
            "decision": log.decision,
            "score": log.score,
            "threshold": log.threshold,
            "challenge_id": log.challenge_id,
            "transcription": log.transcription,
            "sentence_match": log.sentence_match,
            "liveness_passed": log.liveness_passed
        }
        for log in logs
    ]


@app.post("/challenge/generate", response_model=ChallengeResponse)
async def create_challenge(
    user_id: Optional[str] = Form(None),
    complexity: Optional[str] = Form(None)
):
    """
    Generate a new liveness detection challenge.
    
    - **user_id**: Optional user ID for verification challenges
    - **complexity**: Sentence complexity (simple, medium, complex)
    
    Returns a challenge ID and sentence that the user must speak.
    """
    try:
        challenge_id, sentence, expires_at = generate_challenge(
            user_id=user_id,
            complexity=complexity
        )
        return ChallengeResponse(
            challenge_id=challenge_id,
            sentence=sentence,
            expires_at=expires_at.isoformat(),
            user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/challenge/{challenge_id}")
async def get_challenge_details(challenge_id: str):
    """Get challenge details by ID."""
    db = get_database()
    challenge = db.get_challenge(challenge_id)
    
    if challenge is None:
        raise HTTPException(status_code=404, detail=f"Challenge '{challenge_id}' not found")
    
    return {
        "challenge_id": challenge.id,
        "sentence": challenge.sentence,
        "user_id": challenge.user_id,
        "created_at": challenge.created_at.isoformat(),
        "expires_at": challenge.expires_at.isoformat(),
        "used": challenge.used
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
