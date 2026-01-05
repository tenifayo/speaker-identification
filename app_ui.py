"""
Streamlit UI for Speaker Identification System

This application provides a web interface for:
- Enrolling new speakers
- Verifying speakers with liveness detection
- Identifying unknown speakers
- Managing enrolled users
- Viewing access logs
"""

import streamlit as st
import requests
import tempfile
import os
from typing import Optional, List, Dict


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Speaker Identification System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== API Client ====================

class APIClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """Check if API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def enroll_user(self, user_id: str, name: str, audio_files: List[bytes]) -> Optional[Dict]:
        """Enroll a new user with audio samples"""
        try:
            files = [
                ('audio_files', (f'sample_{i}.wav', audio, 'audio/wav'))
                for i, audio in enumerate(audio_files)
            ]
            data = {'user_id': user_id, 'name': name}
            
            response = requests.post(
                f"{self.base_url}/enroll",
                data=data,
                files=files,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Enrollment failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    st.error(f"Details: {e.response.json().get('detail', 'Unknown error')}")
                except:
                    pass
            return None
    
    def generate_challenge(self, user_id: Optional[str] = None) -> Optional[Dict]:
        """Generate a liveness challenge"""
        try:
            data = {}
            if user_id:
                data['user_id'] = user_id
            
            response = requests.post(
                f"{self.base_url}/challenge/generate",
                data=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Challenge generation failed: {str(e)}")
            return None
    
    def verify_speaker(self, user_id: str, audio_file: bytes, challenge_id: Optional[str] = None) -> Optional[Dict]:
        """Verify a speaker"""
        try:
            files = {'audio_file': ('audio.wav', audio_file, 'audio/wav')}
            data = {'user_id': user_id}
            if challenge_id:
                data['challenge_id'] = challenge_id
            
            response = requests.post(
                f"{self.base_url}/verify",
                data=data,
                files=files,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Verification failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    st.error(f"Details: {e.response.json().get('detail', 'Unknown error')}")
                except:
                    pass
            return None
    
    def identify_speaker(self, audio_file: bytes, top_n: int = 5) -> Optional[List[Dict]]:
        """Identify a speaker from audio"""
        try:
            files = {'audio_file': ('audio.wav', audio_file, 'audio/wav')}
            data = {'top_n': top_n}
            
            response = requests.post(
                f"{self.base_url}/identify",
                data=data,
                files=files,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Identification failed: {str(e)}")
            return None
    
    def get_users(self) -> Optional[List[Dict]]:
        """Get all enrolled users"""
        try:
            response = requests.get(f"{self.base_url}/users", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to fetch users: {str(e)}")
            return None
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            response = requests.delete(f"{self.base_url}/users/{user_id}", timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to delete user: {str(e)}")
            return False
    
    def get_logs(self, user_id: Optional[str] = None, limit: int = 100) -> Optional[List[Dict]]:
        """Get access logs"""
        try:
            params = {'limit': limit}
            if user_id:
                params['user_id'] = user_id
            
            response = requests.get(f"{self.base_url}/logs", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to fetch logs: {str(e)}")
            return None

# Initialize API client
api = APIClient()

# ==================== Styling ====================

def apply_custom_css(dark_mode: bool):
    """Apply custom CSS based on theme"""
    if dark_mode:
        css = """
        <style>
        .main {
            background-color: #0F172A;
        }
        .stButton>button {
            background-color: #8B5CF6;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #7C3AED;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        }
        .challenge-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
        }
        .challenge-text {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metric-card {
            background-color: #1E293B;
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #8B5CF6;
        }
        .success-badge {
            background-color: #10B981;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }
        .error-badge {
            background-color: #EF4444;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }
        </style>
        """
    else:
        css = """
        <style>
        .main {
            background-color: #F8FAFC;
        }
        .stButton>button {
            background-color: #7C3AED;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #6D28D9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
        }
        .challenge-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
        }
        .challenge-text {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metric-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #7C3AED;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .success-badge {
            background-color: #10B981;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }
        .error-badge {
            background-color: #EF4444;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# ==================== Session State ====================

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

if 'enrolled_samples' not in st.session_state:
    st.session_state.enrolled_samples = []

if 'current_challenge' not in st.session_state:
    st.session_state.current_challenge = None

# ==================== Sidebar ====================

with st.sidebar:
    st.title("🎙️Speaker Recognition System")
    
    # Theme toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.divider()
    
    # API Status
    st.subheader("API Status")
    if api.health_check():
        st.success("✅ Connected")
    else:
        st.error("❌ Disconnected")
        st.caption(f"API URL: {API_BASE_URL}")
    
    st.divider()
    
    # Navigation
    page = st.pills(
        "Navigation",
        ["📶 Dashboard", "➕ Enroll User", "✅ Verify Speaker", 
         "🔍 Identify Speaker", "👥 User Management", "📋 Access Logs"],
        label_visibility="collapsed",
        selection_mode="single",
        default="📶 Dashboard"
    )

# Apply custom CSS
apply_custom_css(st.session_state.dark_mode)

# ==================== Pages ====================

if page == "📶 Dashboard":
    st.title("📶 Dashboard")
    st.markdown("### Speaker Recognition System Overview")
    
    # Get statistics
    users = api.get_users()
    logs = api.get_logs(limit=10)
    
    if users is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Enrolled Users", len(users))
        
        with col2:
            total_samples = sum(u.get('num_samples', 0) for u in users)
            st.metric("🎙️ Total Samples", total_samples)
        
        with col3:
            if logs:
                granted = sum(1 for log in logs if log.get('decision') == 'granted')
                success_rate = (granted / len(logs) * 100) if logs else 0
                st.metric("✅ Success Rate", f"{success_rate:.1f}%")
            else:
                st.metric("✅ Success Rate", "N/A")
    
    st.divider()
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    if logs:
        for log in logs[:5]:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**User:** {log.get('user_id', 'Unknown')}")
                with col2:
                    st.write(f"**Time:** {log.get('timestamp', 'Unknown')[:19]}")
                with col3:
                    if log.get('decision') == 'granted':
                        st.markdown('<span class="success-badge">Granted</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="error-badge">Denied</span>', unsafe_allow_html=True)
    else:
        st.info("No recent activity")

elif page == "➕ Enroll User":
    st.title("➕ Enroll New User")
    st.markdown("### Record audio samples to enroll a new speaker")
    
    with st.form("enrollment_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.text_input("User ID", placeholder="e.g., user123")
        with col2:
            name = st.text_input("Full Name", placeholder="e.g., John Doe")
        
        st.divider()
        
        st.subheader("🎙️ Audio Samples (Minimum 3)")
        st.caption("Record at least 3 audio samples for better accuracy")
        
        samples = []
        for i in range(5):
            audio = st.audio_input(f"Sample {i+1}" + (" (Required)" if i < 3 else " (Optional)"))
            if audio:
                samples.append(audio.getvalue())
        
        submitted = st.form_submit_button("✅ Enroll User", use_container_width=True)
        
        if submitted:
            if not user_id or not name:
                st.error("❌ Please provide both User ID and Name")
            elif len(samples) < 3:
                st.error(f"❌ Please record at least 3 samples (you have {len(samples)})")
            else:
                with st.spinner("Enrolling user..."):
                    result = api.enroll_user(user_id, name, samples)
                    if result:
                        st.success(f"✅ Successfully enrolled **{result['name']}** with {result['num_samples']} samples!")
                        st.balloons()

elif page == "✅ Verify Speaker":
    st.title("✅ Verify Speaker")
    st.markdown("### Verify speaker identity with liveness detection")
    
    user_id = st.text_input("User ID to Verify", placeholder="e.g., user123")
    
    st.divider()
    
    # Step 1: Generate Challenge
    st.subheader("Step 1: Generate Liveness Challenge")
    if st.button("🎲 Generate Challenge", use_container_width=True):
        with st.spinner("Generating challenge..."):
            challenge = api.generate_challenge(user_id if user_id else None)
            if challenge:
                st.session_state.current_challenge = challenge
                st.success("✅ Challenge generated!")
    
    # Display current challenge
    if st.session_state.current_challenge:
        st.markdown(
            f'<div class="challenge-box"><div class="challenge-text">'
            f'"{st.session_state.current_challenge["sentence"]}"'
            f'</div></div>',
            unsafe_allow_html=True
        )
        st.caption(f"Challenge ID: {st.session_state.current_challenge['challenge_id']}")
        st.caption("⏰ Please speak this sentence clearly")
    
    st.divider()
    
    # Step 2: Record and Verify
    st.subheader("Step 2: Record Your Response")
    audio = st.audio_input("🎙️ Record Audio")
    
    if st.button("🔍 Verify", use_container_width=True, disabled=not audio or not user_id):
        with st.spinner("Verifying..."):
            challenge_id = st.session_state.current_challenge['challenge_id'] if st.session_state.current_challenge else None
            result = api.verify_speaker(user_id, audio.getvalue(), challenge_id)
            
            if result:
                st.divider()
                st.subheader("📊 Verification Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Voice Similarity", f"{result['score']:.2%}")
                with col2:
                    st.metric("Threshold", f"{result['threshold']:.2%}")
                with col3:
                    decision_emoji = "✅" if result['is_verified'] else "❌"
                    st.metric("Decision", f"{decision_emoji} {result['decision'].upper()}")
                
                # Liveness results
                if result.get('liveness_result'):
                    st.divider()
                    st.subheader("🔐 Liveness Check")
                    liveness = result['liveness_result']
                    
                    if liveness.get('passed'):
                        st.success("✅ Liveness check PASSED")
                    else:
                        st.error("❌ Liveness check FAILED")
                    
                    with st.expander("View Details"):
                        st.write(f"**Expected:** {liveness.get('expected_sentence', 'N/A')}")
                        st.write(f"**Transcribed:** {liveness.get('transcribed_text', 'N/A')}")
                        st.write(f"**Similarity:** {liveness.get('similarity_score', 0):.2%}")
                
                # Overall result
                if result['is_verified']:
                    st.success("🎉 **VERIFICATION SUCCESSFUL!**")
                    st.balloons()
                else:
                    st.error("❌ **VERIFICATION FAILED**")

elif page == "🔍 Identify Speaker":
    st.title("🔍 Identify Speaker")
    st.markdown("### Identify an unknown speaker from audio")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        audio = st.audio_input("🎙️ Record Unknown Speaker")
    with col2:
        top_n = st.number_input("Top Matches", min_value=1, max_value=10, value=5)
    
    if st.button("🔍 Identify", use_container_width=True, disabled=not audio):
        with st.spinner("Identifying speaker..."):
            results = api.identify_speaker(audio.getvalue(), top_n)
            
            if results:
                if len(results) == 0:
                    st.warning("⚠️ No matches found above threshold")
                else:
                    st.success(f"✅ Found {len(results)} match(es)")
                    st.divider()
                    
                    for i, match in enumerate(results, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 3, 2])
                            with col1:
                                st.markdown(f"### #{i}")
                            with col2:
                                st.write(f"**User ID:** {match['user_id']}")
                                st.write(f"**Name:** {match['name']}")
                            with col3:
                                score = match['score']
                                st.metric("Similarity", f"{score:.2%}")
                                st.progress(score)
                            st.divider()

elif page == "👥 User Management":
    st.title("👥 User Management")
    st.markdown("### View and manage enrolled users")
    
    # Refresh button
    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()
    
    users = api.get_users()
    
    if users:
        st.info(f"📊 Total Users: **{len(users)}**")
        
        # Search
        search = st.text_input("🔍 Search by User ID or Name", "")
        
        if search:
            users = [u for u in users if search.lower() in u['id'].lower() or search.lower() in u['name'].lower()]
        
        st.divider()
        
        # Display users
        for user in users:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    st.write(f"**ID:** {user['id']}")
                with col2:
                    st.write(f"**Name:** {user['name']}")
                with col3:
                    st.write(f"🎙️ {user['num_samples']} samples")
                with col4:
                    if st.button("🗑️", key=f"del_{user['id']}"):
                        if api.delete_user(user['id']):
                            st.success(f"✅ Deleted {user['id']}")
                            st.rerun()
                
                st.caption(f"Created: {user['created_at'][:10]}")
                st.divider()
    else:
        st.info("No users enrolled yet")

elif page == "📋 Access Logs":
    st.title("📋 Access Logs")
    st.markdown("### View system activity and verification history")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        users = api.get_users()
        user_filter = st.selectbox(
            "Filter by User",
            ["All Users"] + ([u['id'] for u in users] if users else [])
        )
    with col2:
        limit = st.number_input("Max Logs", min_value=10, max_value=500, value=20)
    
    # Fetch logs
    filter_user = None if user_filter == "All Users" else user_filter
    logs = api.get_logs(user_id=filter_user, limit=limit)
    
    if logs:
        st.info(f"📊 Showing {len(logs)} log(s)")
        st.divider()
        
        for log in logs:
            with st.expander(
                f"{'✅' if log['decision'] == 'granted' else '❌'} "
                f"{log['user_id']} - {log['timestamp'][:19]}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**User ID:** {log['user_id']}")
                    st.write(f"**Decision:** {log['decision'].upper()}")
                    st.write(f"**Score:** {log['score']:.2%}")
                with col2:
                    st.write(f"**Threshold:** {log['threshold']:.2%}")
                    st.write(f"**Timestamp:** {log['timestamp']}")
                
                # Liveness info
                if log.get('challenge_id'):
                    st.divider()
                    st.write("**Liveness Detection:**")
                    st.write(f"- Challenge ID: {log['challenge_id']}")
                    st.write(f"- Transcription: {log.get('transcription', 'N/A')}")
                    st.write(f"- Sentence Match: {'✅' if log.get('sentence_match') else '❌'}")
                    st.write(f"- Liveness Passed: {'✅' if log.get('liveness_passed') else '❌'}")
    else:
        st.info("No logs available")

# Footer
st.divider()
st.caption("🎙️ Speaker Recognition System | Built with Streamlit & FastAPI")
