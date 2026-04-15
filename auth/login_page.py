"""
Vibrant Login Page for Text Summarizer
Beautiful, modern authentication UI with OTP-based real-time verification.
"""

import os
import streamlit as st
import time
from dotenv import load_dotenv
from auth import AuthManager
from auth.otp_manager import OTPManager, EmailOTPSender, format_time_remaining

# Load environment variables
load_dotenv()


def get_vibrant_css():
    """Return vibrant CSS styling for login page."""
    return """
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Animated gradient background */
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Login card */
        .login-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            animation: slideUp 0.6s ease-out;
            max-width: 450px;
            margin: 0 auto;
        }

        /* Top brand box shown in the marked area */
        .login-brand-box {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.28), rgba(118, 75, 162, 0.30), rgba(35, 166, 213, 0.28));
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1.5px solid rgba(255, 255, 255, 0.45);
            border-radius: 24px;
            padding: 18px 28px;
            box-shadow: 0 14px 34px rgba(80, 35, 128, 0.28), 0 0 24px rgba(255, 255, 255, 0.12) inset;
            text-align: center;
            max-width: 560px;
            margin: 0 auto 18px auto;
        }

        .brand-title {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #f5eaff 45%, #d7f6ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            text-shadow: 0 2px 18px rgba(20, 8, 38, 0.25);
        }

        .brand-slogan {
            color: #f3efff;
            font-size: 1rem;
            margin-top: 4px;
            font-weight: 600;
            letter-spacing: 0.2px;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Logo and title */
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .login-logo {
            font-size: 4rem;
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .login-title {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-top: 10px;
        }
        
        .login-subtitle {
            color: #666;
            font-size: 1rem;
            margin-top: 5px;
        }
        
        /* Form styling */
        .stTextInput > div > div > input {
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 15px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.32), rgba(255, 255, 255, 0.20));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.45);
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Success/Error messages */
        .success-msg {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            text-align: center;
            animation: fadeIn 0.5s ease;
        }
        
        .error-msg {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            text-align: center;
            animation: shake 0.5s ease;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        
        /* Features section */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 25px;
        }
        
        .feature-item {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .feature-item:hover {
            transform: scale(1.05);
        }
        
        .feature-icon {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        
        .feature-text {
            font-size: 0.85rem;
            color: #666;
            font-weight: 500;
        }
        
        /* Divider */
        .divider {
            display: flex;
            align-items: center;
            margin: 20px 0;
        }
        
        .divider::before, .divider::after {
            content: '';
            flex: 1;
            height: 1px;
            background: #e0e0e0;
        }
        
        .divider-text {
            padding: 0 15px;
            color: #999;
            font-size: 0.9rem;
        }
        
        /* Floating shapes */
        .floating-shapes {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
            pointer-events: none;
        }
        
        .shape {
            position: absolute;
            opacity: 0.1;
            animation: float 20s infinite;
        }
        
        .shape:nth-child(1) { top: 20%; left: 10%; animation-delay: 0s; }
        .shape:nth-child(2) { top: 60%; left: 80%; animation-delay: -5s; }
        .shape:nth-child(3) { top: 40%; left: 50%; animation-delay: -10s; }
        .shape:nth-child(4) { top: 80%; left: 20%; animation-delay: -15s; }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-30px) rotate(180deg); }
        }
        
        /* Password strength indicator */
        .password-strength {
            height: 4px;
            border-radius: 2px;
            margin-top: 5px;
            transition: all 0.3s ease;
        }
        
        .strength-weak { background: linear-gradient(90deg, #eb3349, #f45c43); width: 33%; }
        .strength-medium { background: linear-gradient(90deg, #f7971e, #ffd200); width: 66%; }
        .strength-strong { background: linear-gradient(90deg, #11998e, #38ef7d); width: 100%; }
        
        /* OTP Specific Styles */
        .otp-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }
        
        .otp-input-box {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 2rem;
            letter-spacing: 15px;
            padding: 15px 25px;
            border-radius: 12px;
            text-align: center;
            font-weight: bold;
            font-family: 'Courier New', monospace;
            border: 3px solid rgba(255,255,255,0.3);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4); }
            50% { box-shadow: 0 0 0 15px rgba(102, 126, 234, 0); }
        }
        
        .timer-box {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            margin: 15px auto;
            width: fit-content;
            animation: timerPulse 1s infinite;
        }
        
        @keyframes timerPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        .timer-expired {
            background: linear-gradient(135deg, #bdc3c7, #95a5a6);
            animation: none;
        }
        
        .step-indicator {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .step {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .step-active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            transform: scale(1.1);
        }
        
        .step-completed {
            background: linear-gradient(135deg, #11998e, #38ef7d);
            color: white;
        }
        
        .step-pending {
            background: #e0e0e0;
            color: #999;
        }
        
        .step-line {
            width: 30px;
            height: 3px;
            background: #e0e0e0;
            align-self: center;
        }
        
        .step-line-active {
            background: linear-gradient(90deg, #11998e, #38ef7d);
        }
        
        .resend-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        
        .resend-link:hover {
            color: #764ba2;
        }
        
        .info-banner {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
            border-left: 4px solid #667eea;
            padding: 12px 15px;
            border-radius: 0 10px 10px 0;
            margin: 15px 0;
        }
        
        .info-banner-text {
            color: #555;
            font-size: 0.9rem;
        }
    </style>
    """


def render_login_page():
    """Render the vibrant login page with OTP-based authentication."""
    
    # Initialize auth manager and OTP manager
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    if 'otp_manager' not in st.session_state:
        st.session_state.otp_manager = OTPManager(otp_length=6, expiry_minutes=5, max_attempts=3)
    if 'email_sender' not in st.session_state:
        # Load SMTP settings from environment variables
        st.session_state.email_sender = EmailOTPSender(
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            sender_email=os.getenv("SMTP_EMAIL", ""),
            sender_password=os.getenv("SMTP_PASSWORD", "")
        )
    
    # Initialize OTP login state
    if 'otp_login_step' not in st.session_state:
        st.session_state.otp_login_step = 1  # Step 1: Enter email, Step 2: Enter OTP
    if 'otp_email' not in st.session_state:
        st.session_state.otp_email = ""
    if 'otp_sent_time' not in st.session_state:
        st.session_state.otp_sent_time = None
    
    auth = st.session_state.auth_manager
    otp_mgr = st.session_state.otp_manager
    email_sender = st.session_state.email_sender
    
    # Apply vibrant CSS
    st.markdown(get_vibrant_css(), unsafe_allow_html=True)
    
    # Floating shapes in background
    st.markdown("""
        <div class="floating-shapes">
            <div class="shape" style="font-size: 100px;">📝</div>
            <div class="shape" style="font-size: 80px;">🤖</div>
            <div class="shape" style="font-size: 90px;">✨</div>
            <div class="shape" style="font-size: 70px;">📊</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the login card
    _, col2, _ = st.columns([1, 2, 1])
    
    with col2:
        # Brand box (title + slogan in top highlighted area)
        st.markdown("""
            <div class="login-brand-box">
                <div class="brand-title">✨ Summarix AI</div>
                <div class="brand-slogan">Transforming Information into Insight</div>
            </div>
            <div class="login-header">
                <div class="login-logo">✨</div>
                <div class="login-title">Welcome</div>
                <div class="login-subtitle">Sign in to continue</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Login / Signup tabs
        tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign Up"])
        
        # LOGIN TAB - OTP or Password Based
        with tab1:
            st.markdown("#### Welcome Back!")
            
            # Login method selector
            login_method = st.radio(
                "Choose login method:",
                ["🔑 Password", "📱 OTP (Email)"],
                horizontal=True,
                key="login_method_selector"
            )
            
            # PASSWORD LOGIN
            if login_method == "🔑 Password":
                with st.form("password_login_form", clear_on_submit=False):
                    login_username = st.text_input(
                        "Username or Email",
                        placeholder="Enter your username or email",
                        key="login_user"
                    )
                    
                    login_password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password",
                        key="login_pass"
                    )
                    
                    login_submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                    
                    if login_submitted:
                        if not login_username or not login_password:
                            st.markdown('<div class="error-msg">❌ Please enter username and password</div>', unsafe_allow_html=True)
                        else:
                            success, message, user = auth.login(login_username, login_password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.user = user
                                st.markdown(f'<div class="success-msg">✅ {message}</div>', unsafe_allow_html=True)
                                st.balloons()
                                st.rerun()
                            else:
                                st.markdown(f'<div class="error-msg">❌ {message}</div>', unsafe_allow_html=True)
            
            # OTP LOGIN
            else:
                # Step Indicator
                step = st.session_state.otp_login_step
                step1_class = "step-completed" if step > 1 else ("step-active" if step == 1 else "step-pending")
                step2_class = "step-active" if step == 2 else "step-pending"
                line_class = "step-line-active" if step > 1 else ""
                
                st.markdown(f"""
                    <div class="step-indicator">
                        <div class="step {step1_class}">{'✓' if step > 1 else '1'}</div>
                        <div class="step-line {line_class}"></div>
                        <div class="step {step2_class}">2</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # STEP 1: Enter Email
                if step == 1:
                    st.markdown("""
                        <div class="info-banner">
                            <div class="info-banner-text">📧 Enter your registered email to receive a one-time password</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    email_input = st.text_input(
                        "Email Address",
                        placeholder="Enter your registered email",
                        key="otp_email_input"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📤 Send OTP", use_container_width=True, key="send_otp_btn"):
                            if not email_input:
                                st.markdown('<div class="error-msg">❌ Please enter your email</div>', unsafe_allow_html=True)
                            elif not auth.email_exists(email_input):
                                st.markdown('<div class="error-msg">❌ Email not registered. Please sign up first.</div>', unsafe_allow_html=True)
                            else:
                                # Generate and send OTP
                                otp, _ = otp_mgr.generate_otp(email_input)
                                success, msg = email_sender.send_otp_email(email_input, otp, 5)
                                
                                if success:
                                    st.session_state.otp_email = email_input
                                    st.session_state.otp_sent_time = time.time()
                                    st.session_state.otp_login_step = 2
                                    st.rerun()
                                else:
                                    st.markdown(f'<div class="error-msg">❌ {msg}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        clear_clicked = st.button("🔄 Clear", use_container_width=True, key="clear_email_btn")
                    
                    if clear_clicked:
                        st.session_state.otp_email = ""
                        st.session_state.otp_email_input = ""
                        st.rerun()
                
                # STEP 2: Enter OTP
                elif step == 2:
                    # Back button
                    if st.button("← Back to Email", key="back_to_email_btn"):
                        st.session_state.otp_login_step = 1
                        otp_mgr.clear_otp(st.session_state.otp_email)
                        st.rerun()
                    
                    st.markdown(f"**📧 OTP sent to:** `{st.session_state.otp_email}`")
                    
                    # Real-time countdown timer (display only, no auto-refresh)
                    time_remaining = otp_mgr.get_time_remaining(st.session_state.otp_email)
                    if time_remaining is not None and time_remaining > 0:
                        timer_class = "timer-box"
                        st.markdown(f"""
                            <div class="{timer_class}">
                                ⏱️ OTP expires in: {format_time_remaining(time_remaining)}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="timer-box timer-expired">
                                ⏱️ OTP Expired - Click Resend OTP
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # OTP Input
                    otp_input = st.text_input(
                        "Enter 6-digit OTP",
                        placeholder="Enter the OTP sent to your email",
                        key="otp_verification_input",
                        max_chars=6
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Verify & Login", use_container_width=True, key="verify_otp_btn"):
                            if not otp_input:
                                st.markdown('<div class="error-msg">❌ Please enter OTP</div>', unsafe_allow_html=True)
                            elif len(otp_input) != 6:
                                st.markdown('<div class="error-msg">❌ OTP must be 6 digits</div>', unsafe_allow_html=True)
                            else:
                                # Verify OTP
                                success, msg = otp_mgr.verify_otp(st.session_state.otp_email, otp_input)
                                
                                if success:
                                    # Get user and login
                                    user = auth.get_user_by_email(st.session_state.otp_email)
                                    if user:
                                        st.session_state.logged_in = True
                                        st.session_state.user = user
                                        st.session_state.otp_login_step = 1
                                        otp_mgr.clear_otp(st.session_state.otp_email)
                                        st.markdown(f'<div class="success-msg">✅ {msg}</div>', unsafe_allow_html=True)
                                        st.balloons()
                                        st.rerun()
                                    else:
                                        st.markdown('<div class="error-msg">❌ User not found</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="error-msg">❌ {msg}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("🔄 Resend OTP", use_container_width=True, key="resend_otp_btn"):
                            otp, _ = otp_mgr.generate_otp(st.session_state.otp_email)
                            success, msg = email_sender.send_otp_email(st.session_state.otp_email, otp, 5)
                            if success:
                                st.session_state.otp_sent_time = time.time()
                                st.markdown('<div class="success-msg">✅ New OTP sent to your email!</div>', unsafe_allow_html=True)
                                st.rerun()
                            else:
                                st.markdown(f'<div class="error-msg">❌ {msg}</div>', unsafe_allow_html=True)
        
        # SIGNUP TAB
        with tab2:
            with st.form("signup_form", clear_on_submit=True):
                st.markdown("#### Create Account")
                
                signup_fullname = st.text_input(
                    "Full Name",
                    placeholder="John Doe",
                    key="signup_name"
                )
                
                signup_email = st.text_input(
                    "Email",
                    placeholder="john@example.com",
                    key="signup_email"
                )
                
                signup_username = st.text_input(
                    "Username",
                    placeholder="Choose a username",
                    key="signup_user"
                )
                
                signup_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a strong password",
                    key="signup_pass"
                )
                
                signup_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Confirm your password",
                    key="signup_confirm"
                )
                
                agree_terms = st.checkbox("I agree to the Terms of Service")
                
                signup_submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
                
                if signup_submitted:
                    if not agree_terms:
                        st.markdown('<div class="error-msg">❌ Please agree to the Terms of Service</div>', unsafe_allow_html=True)
                    elif signup_password != signup_confirm:
                        st.markdown('<div class="error-msg">❌ Passwords do not match</div>', unsafe_allow_html=True)
                    else:
                        success, message = auth.register(
                            username=signup_username,
                            email=signup_email,
                            password=signup_password,
                            full_name=signup_fullname
                        )
                        if success:
                            st.markdown(f'<div class="success-msg">✅ {message}</div>', unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f'<div class="error-msg">❌ {message}</div>', unsafe_allow_html=True)
        
        # Features showcase
        st.markdown("""
            <div class="divider">
                <span class="divider-text">Features</span>
            </div>
            <div class="features-grid">
                <div class="feature-item">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-text">AI Summarization</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">📊</div>
                    <div class="feature-text">Multiple Methods</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">📁</div>
                    <div class="feature-text">PDF/DOCX Support</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">📈</div>
                    <div class="feature-text">ROUGE Metrics</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
            <div style="text-align: center; margin-top: 30px; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                <p>🔒 Secure & Private | Developed By Sanket Bhat ❤️ using Streamlit</p>
            </div>
        """, unsafe_allow_html=True)


def check_authentication():
    """
    Check if user is authenticated.
    Returns True if logged in, False otherwise.
    """
    return st.session_state.get('logged_in', False)


def get_current_user():
    """Get the currently logged in user."""
    return st.session_state.get('user', None)


def logout():
    """Log out the current user."""
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


def render_user_menu():
    """Render user menu in sidebar when logged in."""
    user = get_current_user()
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**👤 {user.full_name or user.username}**")
            st.caption(f"📧 {user.email}")
            if st.button("🚪 Logout", use_container_width=True):
                logout()


if __name__ == "__main__":
    # Test the login page
    if not check_authentication():
        render_login_page()
    else:
        st.write("Welcome! You are logged in.")
        render_user_menu()
